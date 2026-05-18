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

GLMakie's default renderloop has `screen.close_after_renderloop = true`. If left
enabled, the renderloop can call `close(screen)` as soon as the native red
button makes `isopen(screen) == false`. That can happen before package frame and
polling timers are stopped.

The current rule is to let GLMakie own the actual screen cleanup, but not the
ordering. Windows-owned screens disable `close_after_renderloop`; the package
close task stops and waits for notification timers first, then calls GLMakie's
official `close(screen; reuse = false)`.

One important API detail from the installed GLMakie source:

```julia
function windowclose(win)
    event[] = false
end
```

The red window button therefore runs every `on(events(fig).window_open)` handler
inside GLMakie's native close callback. Those handlers must not wait, close
Makie objects, or perform process cleanup inline.

## Final Working Fix

The close freeze stopped after treating live plot observables as hot runtime
resources.

The failing pattern was:

```julia
obs = Observable{typeof(vals)}(vals)  # vals points into graph state
image!(axis, obs)
register_frame!(handle) do
    notify(obs)
end
```

For `LayerViewPanel`, `vals` is a view into `state(layer)` for 2D layers, or
`vec(view(...))` for 3D layers. While the window is open, that is exactly what
we want: it avoids per-frame allocation and lets Makie read the current graph
state.

The problem was close teardown. Even after package timers were stopped,
GLMakie's `close(screen)` / `empty!(screen)` still walks plots and cached scene
objects. If a plot observable still stores a view into graph-owned memory that
has been updated by a running process, GLMakie's teardown can freeze,
especially after repeated open/close cycles.

The working rule is:

1. keep the observable concretely typed while live, e.g.
   `Observable{typeof(view(state(layer), :, :))}`;
2. register it with `register_hot_observable!(handle, obs)`;
3. during runtime shutdown, after frame timers stop and before GLMakie screen
   teardown, call `detach_hot_observable!(obs)`;
4. `detach_hot_observable!` dispatches on the observable's type parameter and
   replaces the stored value with a zero-sized inert value of the same type;
5. the replacement is assigned without `notify`, so close does not queue another
   render update.

This means the open window still uses fast live views, but GLMakie teardown no
longer sees plot data pointing into graph state.

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

### Waiting For Timers Inline In `window_open=false`

Even only closing and waiting for the frame/poll timers inside the
`window_open=false` observer made the red button freeze immediately. That
observer is still part of GLFW's close callback, so waiting there blocks
GLMakie's native close path.

Rule: the `window_open=false` observer may mark package state as closing and
schedule cleanup, but it must return immediately.

### Raw GLFW Close From Package Code

Raw GLFW close/hide/destroy calls from package close code caused races with
GLMakie's renderloop and left windows white or contexts dead. Package code
should not drive native close directly; it should flip `window_open` and let
GLMakie complete the backend close.

### Reusing Closed GLMakie Screens

GLMakie keeps a `SCREEN_REUSE_POOL`. The public `GLMakie.Screen()` constructor
pulls from that pool, and `close(screen)` pushes a screen back into it when
`screen.reuse == true`.

The observed symptom was that the first window close usually succeeded, while a
second open/close in the same Julia session was much more likely to freeze.
That matches a reused GL screen inheriting backend state from the previous
interactive window. Windows-owned interactive screens therefore set
`screen.reuse = false` immediately after construction. This still lets GLMakie
own native close, but prevents the closed screen from becoming the next
interactive window.

The window constructor also clears `SCREEN_REUSE_POOL` before creating a new
interactive screen. That handles long-running Julia sessions that may already
contain pooled screens from earlier versions of the close code.

### Letting The Renderloop Auto-Close Before Timers Stop

With `screen.close_after_renderloop = true`, the native close button can make
GLMakie run `close(screen)` / `empty!(screen)` before package timers have
actually been closed. Moving package cleanup to `@async` fixed blocking the
GLFW close callback, but it introduced an ordering race: GLMakie could still
free scene resources while a frame/poll tick was pending.

Rule: disable `close_after_renderloop` for Windows-owned screens and call
`close(screen; reuse = false)` explicitly only after package notification
timers have been stopped and waited.

## Current Approach

For native close:

1. GLMakie or package code sets `host.open[] = false`.
2. A minimal `window_open` observer marks `host.closing = true`, records that
   close cleanup has been scheduled, starts an async cleanup task, and returns.
3. Frame and polling callbacks no-op as soon as `host.closing` is true.
4. The async cleanup task closes and waits for host notification timers, then
   requests owned graph/process shutdown.
5. The async cleanup task calls GLMakie's official `close(screen; reuse = false)`.
6. Host close callbacks are scheduled asynchronously.
7. Before opening a new window, the GLMakie reuse pool is cleared. Windows-owned
   screens are also marked non-reusable, so GLMakie's close path does not put
   them into the reuse pool for the next interface.
8. Panel handles are marked closed so package callbacks stop calling them, but their
   `close!` hooks, Makie contents, observer resources, and arbitrary resources
   are not touched by full-window close.

For explicit close:

1. `close(host)` sets `host.open[] = false` for displayed GLMakie windows.
2. The same package cleanup observer path runs as for the native window button.

For Cmd+W:

1. The keyboard callback sets `host.open[] = false`, so package cleanup is the
   same as the native close path.
2. It then schedules GLMakie's official `close(screen)` asynchronously. Cmd+W
   is not a native OS close event, so it cannot be literally identical to the
   red button at the GLFW level.

For graph/process cleanup:

- graph panels register one host-level cleanup per graph identity;
- cleanup calls `_request_graph_process_close!(g)`;
- that function requests each process stop with `Processes.shouldrun(process, false)`,
  empties `processes(g)`, and reaps process state later;
- it does not wait for process tasks before returning.

## Close Debugging Harness

The current close issue is being narrowed with:

```julia
include("examples/WindowCloseDebugging.jl")
```

The harness has three tiers.

### Facility Windows

Use `open_next!()` to walk through progressively larger stripped-down windows:

- raw `GLMakie.Screen` with a label;
- `WindowHost` with no timers;
- `WindowHost` with frame/poll timers;
- simple frame and polled labels;
- static layer drawing;
- individual panels;
- `SimulationPanel`;
- exact public `interface(g)`.

These windows test the framework and panel facilities against a small toy graph.
As of the latest manual pass, none of these froze the Julia REPL, and none
reproduced the original hard close freeze. Some native windows can still take a
while to disappear, which points at GLMakie/native teardown rather than a Julia
task deadlock.

### Generic Process Windows

Use:

```julia
open_process_debug!(:simulation_open_then_langevin)
open_process_debug!(:public_interface_open_then_langevin)
```

These test the same toy graph with a running process, including the important
ordering used by examples:

```julia
host = interface(g)
createProcess(g, algorithm)
```

As of the latest manual pass, generic process variants also did not reproduce
the freeze. That means a running process by itself is not enough.

### Example-Like Scenarios

Use:

```julia
open_scenario_debug!(:example_3d_langevin_coulomb)
```

Available scenarios are printed by the example in `CLOSE_DEBUG_SCENARIOS`.
They construct graph/algorithm pairs based on the real example files, then open
the public `interface(g)` and start the process afterwards. These are the next
suspects because the simplified facility and generic-process windows do not
reproduce the issue.

When reporting a close result, record:

- the debug name;
- whether the graph visibly updates before close;
- whether the Julia REPL stays responsive;
- whether the native window disappears, lingers, or turns white;
- whether the issue appears only after repeated open/close cycles.

### Reproducer Notes: Simulation Composite With Langevin

The first debug variant reported to reproduce the bad close behavior was:

```julia
open_process_debug!(:public_interface_running_langevin)
```

Changing public `new_interface` to mount after display was tried because the
closely related `simulation_running_langevin` variant initially appeared not to
reproduce the freeze. Later manual testing showed that the simulation composite
can still reproduce the bad close behavior, so mount/display order is not a
complete explanation.

The current active diagnostic path is to test subset composites:

```julia
open_next_suspect!()
```

The suspect sequence is intentionally short and focused around the smallest
reported reproducer neighborhood:

```julia
CLOSE_DEBUG_SUSPECT_NAMES
```

It currently covers:

- `LayerViewPanel` only;
- `TemperaturePanel` only;
- `KineticTimePanel` only;
- `LayerViewPanel + TemperaturePanel`;
- `LayerViewPanel + KineticTimePanel`;
- `LayerViewPanel + KineticTimePanel + TemperaturePanel`.

These keep a running graph process, defaulting to `LocalLangevin` in the helper
sequence, but vary only the suspected panel contents. The goal is to
distinguish:

- a pairwise interaction between subpanels, e.g. status counters plus layer
  frame notifications;
- a specific full-window resource, e.g. temperature slider or Hamiltonian
  parameter display;
- a pure GLMakie complexity threshold from many widgets/plots in one figure.

### Current Smallest Reproducer

As of the latest manual pass on 2026-05-16,
`subset_layer_only_langevin` did reproduce a complete close freeze.
`subset_layer_kinetic_langevin` also lingered for a while before eventually
recovering in one pass.

Repeat-cycle observation: opening and closing `subset_layer_only_langevin`
twice succeeded, then the third open/close froze. That makes accumulated
GLMakie/window/backend state across repeated opens a live suspect, not just the
contents of a single panel instance.

The current smallest reported reproducer is:

```julia
open_suspect_debug!(:subset_layer_only_langevin)
```

Despite the historical name, do not treat this as a specifically
`LocalLangevin` bug yet. The important condition is currently:

- one `LayerViewPanel`;
- a graph process mutating `g`;
- the host frame timer notifying the layer plot.

The same debug windows can be run with another graph process algorithm:

```julia
open_suspect_debug!(:subset_layer_only_langevin; process_algorithm = nothing)
```

Passing `process_algorithm = nothing` uses the package default process
constructor. Passing a concrete algorithm tests the same window against that
algorithm.

The focused suspect sequence is:

```julia
open_next_suspect!()
```

and includes:

- `subset_layer_only_langevin`;
- `subset_layer_only_empty_on_close_langevin`;
- `subset_layer_only_copy_langevin`;
- `subset_layer_only_no_polltimer_langevin`;
- `subset_temperature_only_langevin`;
- `subset_kinetic_only_langevin`;
- `subset_layer_temperature_no_kinetic_langevin`;
- `subset_layer_kinetic_langevin`;
- `subset_layer_temperature_langevin`.

`subset_layer_only_langevin` contains:

- `WindowHost`;
- the host frame and poll timers;
- one `LayerViewPanel`;
- a running graph process, defaulting to `LocalLangevin` in the debug sequence.

It does not mount `StatusPanel`, `TemperaturePanel`, `KineticTimePanel`,
`MagnetizationPanel`, or `HamiltonianParameterPanel`.

It also does not register any panel `PolledObservable`. Therefore a
panel-owned `PolledObservable` is unlikely to be the direct trigger for this
particular reproducer. The host still has its generic poll timer, but the
`pollables` list should be empty in this variant.

To test the host poll timer separately, use:

```julia
open_suspect_debug!(:subset_layer_only_no_polltimer_langevin)
```

That keeps `LayerViewPanel` and a running graph process, but disables the host
poll timer. This variant must disable timer startup with
`start_poll_timer = false`; do not use `polling_rate = 0`, because that tests a
broken timer configuration rather than the close lifecycle.

To test whether copying graph state away from Makie's live plot data fixes the
minimal reproducer, use:

```julia
open_suspect_debug!(:subset_layer_only_copy_langevin)
```

That variant keeps the same running graph process and host frame timer, but
replaces `LayerViewPanel` with a debug-only copied layer panel. The Makie
observable owns its display buffer, and each frame copies current graph state
into that buffer before `notify`.

To test whether merely severing the observable's reference to graph state during
shutdown fixes the minimal reproducer, use:

```julia
open_suspect_debug!(:subset_layer_only_empty_on_close_langevin)
```

That variant keeps a live view-backed layer plot while the window is open. When
the host close path stops timers and shuts down panel runtime resources, it
mutates the plot observable's stored value to an empty array without notifying
Makie. If this fixes the repeated-close freeze, the failing edge is likely
GLMakie teardown touching plot observable memory that still points into a graph
being mutated by a process.

Manual result: opening and closing
`subset_layer_only_empty_on_close_langevin` repeatedly did not freeze. The same
observable-detach behavior has therefore been generalized as
`register_hot_observable!`. `LayerViewPanel` keeps concretely typed observables
such as `Observable{typeof(view(state(layer), :, :))}` while the window is live.
During runtime shutdown, `detach_hot_observable!` dispatches on the observable's
type parameter and replaces the stored value with a zero-sized inert value of
the same observable value type. The replacement is done without `notify`, so
close does not enqueue another render update.

### Layer Display Memory Attempts That Did Not Fix It

Two live `LayerViewPanel` memory strategies were tried and did not fix the
freeze:

1. Plain Julia views:

   - 2D display used `Observable(view(state(layer), :, :))`;
   - 3D display used `Observable(vec(view(state(layer), :, :, :)))`;
   - frame callback only called `notify(img_obs)`.

2. Owned display buffer:

   - 2D display allocated `copy(state(layer))` on redraw/layer switch;
   - 3D display allocated a `Float64` vector on redraw/layer switch;
   - each frame copied graph state into the existing display buffer, then
     notified Makie;
   - no locks were added.

Both strategies have been tested against `subset_layer_only_langevin` across
manual passes. Interpret failures as "LayerViewPanel plus a running graph
process" unless a later pass shows that changing only the process algorithm
changes the outcome.

Current conclusion:

- copying the layer state into Makie-owned display buffers did not reliably
  remove the historical close freeze;
- plain Julia `view`/`vec(view(...))` display data also did not reliably remove
  the historical close freeze;
- adding locks was intentionally not tried and should not be used as the next
  default workaround;
- the freeze occurred with only `LayerViewPanel` plus a running graph process,
  so the left Hamiltonian buttons, temperature slider, magnetization readout,
  status counters, and full `SimulationPanel` layout are not required for the
  current smallest reproducer;
- panel-owned `PolledObservable`s are not present in the smallest current
  subset, and the latest no-poll-timer suspect pass did not completely freeze.
- the fix that helped was not copying while live. It was keeping live views
  during normal rendering, then detaching hot observables from graph memory with
  same-type zero-sized replacements during close cleanup.

Keep this earlier mount-order observation in mind, but do not treat it as the
root cause by itself:

- `simulation_running_langevin` opened a `WindowHost` / GLMakie screen first,
  then mounted `SimulationPanel`;
- the public `new_interface` path mounted `SimulationPanel` into an undisplayed
  `Figure`, then displayed the GLMakie screen.

`new_interface` now follows the debug variant order:

```julia
host = WindowHost(...)
_display_host!(host, title)
host[:simulation] = panel!(host, :simulation, SimulationPanel(g), (1, 1))
```

If close behavior regresses again, keep this distinction in the debug matrix:
pre-display panel mounting can exercise a different GLMakie scene attachment
and cleanup path than post-display mounting.

## Rules For Future Changes

- Do not call `close(host)` directly from Makie/GLMakie input callbacks such as
  Cmd+W keyboard handlers. Those callbacks run inside GLMakie's event/renderloop
  handling and can close re-entrantly. Flip `host.open[] = false` and let the
  `window_open` observer perform the same close path as the native window button.
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
- Do not wait for timers inside the `window_open=false` observer. Mark the host
  closing, schedule timer shutdown on a task, and return.
- Do not use raw GLFW close/hide/destroy calls for normal interactive window
  shutdown from package code.
- Do not leave `close_after_renderloop` enabled for Windows-owned screens unless
  notification timers are stopped before the renderloop can enter
  `close(screen)`.
- Do not leave interactive Windows-owned screens reusable unless the repeated
  open/close freeze is known to be fixed upstream.
- If adding a new panel that owns a graph, register graph cleanup through the
  existing `_register_graph_close!` helper.
- If adding a new panel that owns a process directly, register process cleanup
  through `_register_process_close!` or an `onclose!` callback that does not wait.
