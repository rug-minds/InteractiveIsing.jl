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

GLMakie's own `close(screen)` implementation already:

- sets `screen.window_open[] = false`;
- stops the renderloop;
- empties the screen;
- asks GLFW to close the native window.

Therefore `WindowHost` should not try to manually destroy Makie objects during
the `window_open=false` callback. That callback happens in the middle of
GLMakie's own close sequence.

## Things Tried That Were Bad

### Closing Makie Objects Explicitly

Deleting axes, plots, or layout contents during window close caused or worsened
hangs. Makie/GLMakie should own cleanup of its scene graph on close.

Rule: panel close during native window close should not call Makie deletion APIs.

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

Rule: the native close observer should do the minimum needed to stop package
timers/callbacks, then schedule package cleanup asynchronously.

### Raw `GLFW.SetWindowShouldClose` For Explicit Close

Using raw GLFW close requests bypasses some of GLMakie's own close logic. For an
explicit `close(host)` on a real GL window, prefer `close(host.screen)` so
GLMakie runs its normal close sequence.

## Current Approach

For native close:

1. The package does not register a cleanup observer on `window_open`.
2. The frame/poll timer treats `window_open` as state. When it sees
   `host.open[] == false`, it calls `_schedule_native_close!(host)`.
3. `_begin_native_close!(host)` marks the host closing/closed, closes the frame
   and polling timers, and clears frame/poll callback registries.
4. The rest of package cleanup is scheduled with `@async _finish_native_close!(host)`.
5. `_finish_native_close!` closes panel handles and runs registered close
   callbacks, but uses a native cleanup path that avoids generic Makie object
   deletion.

For explicit close:

1. If `host.screen` exists, `close(host)` first starts the same nonblocking host
   cleanup sequence.
2. It then calls `close(host.screen)`, letting GLMakie own the actual window
   shutdown.

For graph/process cleanup:

- graph panels register one host-level cleanup per graph identity;
- cleanup calls `_request_graph_process_close!(g)`;
- that function requests each process stop with `Processes.shouldrun(process, false)`,
  empties `processes(g)`, and reaps process state later;
- it does not wait for process tasks before returning.

## Rules For Future Changes

- Do not call `Processes.close(g)` from a native close callback.
- Do not call `close(process)` from a native close callback unless it is wrapped
  in a task and cannot block the close event.
- Prefer nonblocking stop requests for graph-attached processes.
- Do not delete Makie plots, axes, layout blocks, or scenes as part of native
  close cleanup.
- Do not attach substantial package cleanup directly to `window_open=false`.
  Prefer reading `window_open` as state from package-owned timers.
- Use `close(host.screen)` for explicit real-window close, not raw GLFW close
  calls.
- If adding a new panel that owns a graph, register graph cleanup through the
  existing `_register_graph_close!` helper.
- If adding a new panel that owns a process directly, register process cleanup
  through `_register_process_close!` or an `onclose!` callback that does not wait.
