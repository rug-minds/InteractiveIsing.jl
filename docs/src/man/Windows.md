# Windows

The Windows interface is the GLMakie-based frontend for interactive Ising graph
inspection. The default entry point is:

```julia
using InteractiveIsing

g = IsingGraph(64, 64, Continuous())
host = interface(g)
```

`interface(g)` returns a `WindowHost`. Keeping the return value is useful when
you want to close the window from code:

```julia
close(host)
```

The standard window accepts layout options through keyword arguments. To keep
the central graph/field display but hide the left Hamiltonian selector buttons:

```julia
host = interface(g; hide_left_buttons = true)
```

## Default Simulation Interface

The default interface contains:

- a top status strip with steps/sec, process pause/resume, camera, and
  correlation controls;
- a left selector for graph state and displayable Hamiltonian fields;
- a central layer display;
- a temperature slider that stays synchronized with `temp(g)`;
- bottom magnetization and defect controls.

With `hide_left_buttons = true`, the left selector is collapsed and the central
display starts on the graph state. This is useful when you want the standard
simulation controls but do not need to switch Hamiltonian fields from the UI.

The pause button controls graph processes only. It does not pause the window
frame or polling timers, so the display continues to update while you inspect or
modify the graph from the REPL.

Temperature and process pause state are polled from the graph. This means calls
such as:

```julia
temp!(g, 2.0)
pause.(processes(g))
```

are reflected in the interface without needing to go through the UI controls.

## Hamiltonian Display Entries

The Hamiltonian selector shows the graph state plus displayable Hamiltonian
quantities. For ordinary parameter-template Hamiltonians, graph-sized vector
parameters are detected automatically. Scalar parameters and non-state-sized
arrays are ignored.

If a Hamiltonian should show only a subset of compatible parameters, extend:

```julia
InteractiveIsing.Windows.displayable_hamiltonian_parameters(term::MyTerm, g) =
    (:field, :bias)
```

For a fully custom derived layer display, extend `hamiltonian_visualizations`:

```julia
function InteractiveIsing.Windows.hamiltonian_visualizations(term::MyTerm, g)
    return [
        InteractiveIsing.Windows.layer_display(
            :local_energy,
            layer -> local_energy_array(term, layer);
            colormap = :viridis,
            colorrange = :data,
        ),
    ]
end
```

For a single graph-sized parameter, use `parameter_display`:

```julia
function InteractiveIsing.Windows.hamiltonian_visualizations(term::MyTerm, g)
    return [
        InteractiveIsing.Windows.parameter_display(term, :field, g;
            colormap = :thermal,
            colorrange = :data,
        ),
    ]
end
```

## Opening Bare Windows

The lower-level constructor is:

```julia
host = InteractiveIsing.Windows.window(
    title = "My Window",
    size = (1000, 800),
    fps = 30,
    polling_rate = 10,
)
```

Panels are mounted into the window with `panel!`:

```julia
using InteractiveIsing.Windows

host = window()
handle = panel!(host, SimulationPanel(g), (1, 1))
```

`panel!` also works on a parent `PanelHandle`, which lets composite panels own
subpanels.

## Close Debugging

If GLMakie close behavior regresses, use the close-debug example instead of
changing the close path by guesswork:

```julia
include("examples/WindowCloseDebugging.jl")
```

Walk through stripped-down facility windows:

```julia
open_next!()
```

After each call, close the native window manually and record whether the REPL
stays responsive and whether the native window disappears cleanly.

For process-specific close paths, use:

```julia
open_process_debug!(:public_interface_open_then_langevin)
open_process_debug!(:simulation_open_then_langevin)
```

If the full simulation composite reproduces but individual panels do not, use
the focused suspect sequence:

```julia
open_next_suspect!()
```

The current suspect sequence covers only `LayerViewPanel`, `TemperaturePanel`,
and `KineticTimePanel` combinations.

Current debugging notes:

- `subset_layer_only_langevin` is the smallest reported reproducer.
- That variant does not mount a panel-owned `PolledObservable`.
- `subset_layer_only_no_polltimer_langevin` disables the host poll timer for
  the same reproducer neighborhood.
- Plain Julia views for the layer state did not fix the freeze.
- An owned display buffer with per-frame `copyto!` also did not fix the freeze.
- `subset_layer_only_empty_on_close_langevin` did not freeze in repeated manual
  open/close testing. The production `LayerViewPanel` now registers its live
  plot observable as a hot observable so close cleanup swaps the stored value to
  a zero-sized inert replacement without notifying Makie.

For real example-like graph/algorithm pairs, use:

```julia
open_scenario_debug!(:example_3d_langevin_coulomb)
```

The scenario list is printed as `CLOSE_DEBUG_SCENARIOS` when the example is
loaded. The key distinction is that the generic facility windows use a small
toy graph, while scenario windows build the larger graph/algorithm combinations
from the example files and then run:

```julia
host = interface(g)
createProcess(g, algorithm)
```

## Dynamic Context Lines

`InteractiveLinesPanel` plots two dynamic containers against each other. The
basic constructor accepts two vector-like containers. A getter constructor can
return the current `(x, y)` containers, and `ContextLinesPanel` is the
context-variable entry point on top of the same panel. If the two containers
briefly have different lengths, the plot uses the shortest current length.

```julia
using InteractiveIsing.Windows

host = window(title = "Context Trace")
panel!(
    host,
    ContextLinesPanel(
        process,
        :Metropolis_1 => :accepted_steps,
        :Metropolis_1 => :energy;
        xlabel = "Accepted steps",
        ylabel = "Energy",
        title = "Energy trace",
        line_kwargs = (; color = :dodgerblue),
    ),
)
```

## Exporting Images

Windows distinguish between a literal screenshot and a minimal data export.
Use `fullimage` when you want the actual current Makie window, including UI
controls:

```julia
fullimage("window.png", host)
```

Use `toimage` when you want an export-oriented representation. `toimage` builds
a fresh figure from the useful panel content first, omitting controls such as
buttons and sliders where the panel provides a better data-only view:

```julia
toimage("simulation.png", host)
toimage("connections.png", handle)
```

Axis-hosting panels can also export just their mounted axis:

```julia
axis_to_png("axis.png", handle)
```

For custom panels, the simplest export interface is to expose an axis:

```julia
InteractiveIsing.Windows.axis_trait(::Type{MyPanel}) = HasAxis()
```

If the panel should build a cleaner export than its mounted UI, implement an
image builder:

```julia
InteractiveIsing.Windows.image_trait(::Type{MyPanel}) = HasImage()

function InteractiveIsing.Windows.toimage!(cell, panel::MyPanel, handle; kwargs...)
    ax = Axis(cell; title = "Export")
    lines!(ax, handle[:x][], handle[:y][])
    return ax
end
```

## Connection Graphs

`ConnectionsPanel` renders the graph adjacency as line segments between lattice
sites. It is useful as a quick structural check for generated adjacency
patterns.

```julia
using InteractiveIsing.Windows

host = window(title = "Connections")
panel!(
    host,
    ConnectionsPanel(g;
        selected_nodes = [(1, (10, 10)), (1, (30, 30))],
        selection_mode = :incident,
        max_edges = 20_000,
        curved = true,
        curve_amount = 0.12,
        colormap = :viridis,
        line_kwargs = (; linewidth = 1),
        node_kwargs = (; markersize = 4),
    ),
)
```

## All-Layer Layouts

`AllLayersViewPanel` draws every positioned 2D layer into one shared axis. It
uses `coords(layer)` as a global `(y, x, z)` layer origin and plots each layer
at `[x, x + width] x [y, y + height]`. The panel currently supports only 2D
layers. Layers need explicit, unique xy coordinates; duplicate or overlapping
positions throw an error.

```julia
using InteractiveIsing.Windows

g = IsingGraph(
    Layer(32, 32, Continuous(), Coords(y = 0, x = 0, z = 0)),
    Layer(32, 32, Continuous(), Coords(y = 0, x = 32, z = 0)),
)

host = window(title = "All layers")
panel!(host, AllLayersViewPanel(g; initial_view = :all), (1, 1))
```

The resulting axis uses Makie's normal drag-pan and scroll-zoom interactions,
so it works as one large scrollable map of layer state.

## Close Callbacks

Use `onclose!` for cleanup that should run when a host or panel closes:

```julia
onclose!(host) do host
    @info "window closed"
end
```

The callback is scheduled asynchronously, so it does not block the GLMakie
window close event. Built-in graph panels use this to close graph-attached
processes by requesting that they stop without waiting in the close callback.

## Hot Observables

Use `register_hot_observable!` for high-frequency Makie observables whose
stored value may point into simulation-owned memory, for example a plot
observable containing `view(state(layer), :, :)`.

```julia
vals = view(state(layer), :, :)
obs = Observable{typeof(vals)}(vals)
register_hot_observable!(handle, obs)
```

On host or panel runtime shutdown, `detach_hot_observable!` replaces the stored
observable value with `hot_observable_zero(typeof(vals))` without calling
`notify`. The default implementation dispatches on the observable value type and
currently handles normal arrays, full-slice `SubArray`s, and `vec(view(...))`
reshaped arrays. This keeps the observable concretely typed while preventing
GLMakie close teardown from reading graph memory that may have been updated by
a process.

## Public API

```@docs
InteractiveIsing.Windows.interface
InteractiveIsing.Windows.new_interface
InteractiveIsing.Windows.window
InteractiveIsing.Windows.WindowHost
InteractiveIsing.Windows.AbstractPanel
InteractiveIsing.Windows.PanelHandle
InteractiveIsing.Windows.panel!
InteractiveIsing.Windows.mount!
InteractiveIsing.Windows.register!
InteractiveIsing.Windows.onclose!
InteractiveIsing.Windows.register_frame!
InteractiveIsing.Windows.register_hot_observable!
InteractiveIsing.Windows.hot_observable_zero
InteractiveIsing.Windows.detach_hot_observable!
InteractiveIsing.Windows.register_polled!
InteractiveIsing.Windows.pause!
InteractiveIsing.Windows.resume!
InteractiveIsing.Windows.restart!
InteractiveIsing.Windows.AxisTrait
InteractiveIsing.Windows.HasAxis
InteractiveIsing.Windows.NoAxis
InteractiveIsing.Windows.ImageTrait
InteractiveIsing.Windows.HasImage
InteractiveIsing.Windows.NoImage
InteractiveIsing.Windows.axis_trait
InteractiveIsing.Windows.axiskey
InteractiveIsing.Windows.image_trait
InteractiveIsing.Windows.hasaxis
InteractiveIsing.Windows.hasimage
InteractiveIsing.Windows.getaxis
InteractiveIsing.Windows.axis_to_png
InteractiveIsing.Windows.tofigure
InteractiveIsing.Windows.toimage!
InteractiveIsing.Windows.toimage
InteractiveIsing.Windows.fullimage
InteractiveIsing.Windows.SimulationPanel
InteractiveIsing.Windows.StatusPanel
InteractiveIsing.Windows.LayerSelectorPanel
InteractiveIsing.Windows.LayerViewPanel
InteractiveIsing.Windows.AllLayersViewPanel
InteractiveIsing.Windows.TemperaturePanel
InteractiveIsing.Windows.MagnetizationPanel
InteractiveIsing.Windows.HamiltonianParameterPanel
InteractiveIsing.Windows.InteractiveLinesPanel
InteractiveIsing.Windows.ContextLinesPanel
InteractiveIsing.Windows.ConnectionsPanel
InteractiveIsing.Windows.HamiltonianDisplaySpec
InteractiveIsing.Windows.LayerDisplayValue
InteractiveIsing.Windows.displayable_hamiltonian_parameters
InteractiveIsing.Windows.hamiltonian_visualizations
InteractiveIsing.Windows.parameter_display
InteractiveIsing.Windows.layer_display
```
