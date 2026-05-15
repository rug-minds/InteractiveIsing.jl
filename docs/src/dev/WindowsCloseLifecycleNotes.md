# Windows Close Lifecycle Notes

This note records what has already been tried for the GLMakie Windows close
hangs, so future changes do not repeat the same mistakes.

## Problem

Closing a GLMakie window can freeze or pause Julia for a few seconds. The
freeze has shown up most clearly when an interactive simulation window owns an
`IsingGraph` with running `Processes.Process` tasks.

There are two separate close paths:

- native close: the user clicks the platform window close button, which causes
  GLMakie to update `events(fig).window_open` / `screen.window_open`;
- explicit close: package code calls `close(host)`.

The native close path is especially sensitive because it runs inside GLMakie's
own window/renderloop lifecycle.

## GLMakie Facts Checked

The current pattern for separate windows is still:

```julia
screen = GLMakie.Screen()
display(screen, fig)
```

GLMakie's own `close(screen)` implementation:

- sets `screen.window_open[] = false`;
- stops the renderloop;
- empties the screen;
- asks GLFW to close the native window.

GLMakie's default renderloop also has `screen.close_after_renderloop = true`.
When the native close button makes `isopen(screen) == false`, the renderloop
exits and then calls `close(screen)` itself. For complex interactive figures,
that means the native close button can still run full `empty!(screen)` cleanup
even if package code avoids all Makie deletion. This was enough to cause close
freezes even when graph processes were already paused.

Therefore `WindowHost` disables `close_after_renderloop` for Windows-owned
screens and does not call `close(screen)` for normal interactive close. The
host stops package timers and owned processes, hides the native GLFW window,
marks it should-close, and detaches the screen from GLMakie's `closeall`
registry so the same expensive cleanup is not retried at Julia exit. The GLFW
context is deliberately not destroyed during this close path; destroying it can
race GLMakie's renderloop and produce `Context is not alive anymore!`.
The hide is deferred until the renderloop task has exited, and the close path
does not call `GLFW.PollEvents()` from package timers because that can race
GLMakie's own event polling.

## Things Tried That Were Bad

### Example-Level `window_open` Cleanup

`examples/XORInteractiveLearning.jl` still had its own
`on(host.open) do isopen` observer after the generic window lifecycle had been
fixed. That observer called `Processes.close(graph)` and trainer cleanup when
the native window set `window_open=false`, which reintroduced the same close
freeze outside `src/Windows`.

Rule: examples should use `onclose!(host)` and nonblocking stop requests too;
they should not observe `host.open` directly for cleanup.

### Closing Makie Objects Explicitly

Deleting axes, plots, or layout contents during window close caused or worsened
hangs. Makie/GLMakie should own cleanup of its scene graph on close.

Rule: panel close during native window close should not call Makie deletion APIs.
For interactive Windows-owned GL screens, even GLMakie's own `empty!(screen)` is
avoided on the close path because it can block for seconds.

### Blocking Process Close In The Window Callback

Calling `Processes.close(g)` or `close(process)` directly from close hooks is
bad for interactive windows. `Processes.close(process)` waits for the process
task, and `Processes.close(g)` does that for every graph-attached process.

This can block the close callback or occupy Julia while GLMakie is trying to
finish its own close/renderloop shutdown.

Rule: window close should request process shutdown, not wait for it.

### Doing Full Panel Cleanup Inline In `window_open=false`

Running child close hooks, observer cleanup, resource cleanup, and graph/process
cleanup directly from the native `window_open=false` observer is too much work
inside GLMakie's close path.

Rule: host/window close should not recurse through panel handles. It should stop
package timers and owned processes only. Direct `close(handle)` is still the API
for explicitly destroying an individual mounted panel.

### Calling Raw GLFW Without Disabling GLMakie Auto-Close

Using raw GLFW close requests while leaving `screen.close_after_renderloop =
true` does not fix the problem: the renderloop can still observe the native
close and call `close(screen)` itself. Raw GLFW close requests are only useful
after disabling GLMakie's automatic renderloop close for that screen.

## Current Approach

For native close:

1. The package does not register a cleanup observer on `window_open`.
2. The frame/poll timer treats `window_open` as state. When it sees
   `host.open[] == false`, it calls `_schedule_native_close!(host)`.
3. `_begin_native_close!(host)` marks the host closing/closed, closes the frame
   and polling timers, and clears frame/poll callback registries.
4. Host close callbacks are scheduled asynchronously.
5. Panel handles are marked closed so package timers stop calling them, but their
   `close!` hooks, Makie contents, observer resources, and arbitrary resources
   are not touched by full-window close.

For explicit close:

1. If `host.screen` exists, `close(host)` first starts the same nonblocking host
   cleanup sequence.
2. It then sets `events(fig).window_open[] = false`, marks the GLFW window
   should-close, defers hiding until the renderloop exits, and prevents GLMakie's
   renderloop from calling `close(screen)` / `empty!(screen)`.

For graph/process cleanup:

- graph panels register one host-level cleanup per graph identity;
- cleanup calls `_request_graph_process_close!(g)`;
- that function requests each process stop with `Processes.shouldrun(process, false)`,
  empties `processes(g)`, and reaps process state later;
- it does not wait for process tasks before returning.

## Rules For Future Changes

- Do not call `close(host)` directly from Makie/GLMakie input callbacks such as
  Cmd+W keyboard handlers. Those callbacks run inside GLMakie's event/renderloop
  handling and can close re-entrantly. Flip `host.open[] = false` and let the
  host timer perform the same deferred close path as the native window button.
- Do not call `Processes.close(g)` from a native close callback.
- Do not call `close(process)` from a native close callback unless it is wrapped
  in a task and cannot block the close event.
- Prefer nonblocking stop requests for graph-attached processes.
- Do not delete Makie plots, axes, layout blocks, or scenes as part of native
  close cleanup.
- Do not close panel handles from host/window close. Panel close is for explicit
  panel teardown, not whole-window teardown.
- Do not attach substantial package cleanup directly to `window_open=false`.
  Prefer reading `window_open` as state from package-owned timers.
- Do not use `close(host.screen)` for normal interactive window shutdown. It
  runs `empty!(screen)`, which is the freeze-prone cleanup path.
- If adding a new panel that owns a graph, register graph cleanup through the
  existing `_register_graph_close!` helper.
- If adding a new panel that owns a process directly, register process cleanup
  through `_register_process_close!` or an `onclose!` callback that does not wait.
