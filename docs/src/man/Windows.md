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

## Default Simulation Interface

The default interface contains:

- a top status strip with steps/sec, process pause/resume, camera, and
  correlation controls;
- a left selector for graph state and displayable Hamiltonian fields;
- a central layer display;
- a temperature slider that stays synchronized with `temp(g)`;
- bottom magnetization and defect controls.

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

## Dynamic Context Lines

`ContextLinesPanel` plots two context containers against each other and refreshes
them from the context every frame. If the two containers briefly have different
lengths, the plot uses the shortest current length.

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
InteractiveIsing.Windows.register_frame!
InteractiveIsing.Windows.register_polled!
InteractiveIsing.Windows.pause!
InteractiveIsing.Windows.resume!
InteractiveIsing.Windows.restart!
InteractiveIsing.Windows.SimulationPanel
InteractiveIsing.Windows.StatusPanel
InteractiveIsing.Windows.LayerSelectorPanel
InteractiveIsing.Windows.LayerViewPanel
InteractiveIsing.Windows.TemperaturePanel
InteractiveIsing.Windows.MagnetizationPanel
InteractiveIsing.Windows.HamiltonianParameterPanel
InteractiveIsing.Windows.ContextLinesPanel
InteractiveIsing.Windows.ConnectionsPanel
InteractiveIsing.Windows.HamiltonianDisplaySpec
InteractiveIsing.Windows.LayerDisplayValue
InteractiveIsing.Windows.displayable_hamiltonian_parameters
InteractiveIsing.Windows.hamiltonian_visualizations
InteractiveIsing.Windows.parameter_display
InteractiveIsing.Windows.layer_display
```
