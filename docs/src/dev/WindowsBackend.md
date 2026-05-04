# Windows Backend

The Windows code is a small panel framework on top of GLMakie. The goal is to
make interactive graph UIs composable without forcing every panel to share one
large global state object.

## Host And Panel Handles

`WindowHost` is the root owner. It stores:

- the Makie `Figure`;
- the optional `GLMakie.Screen`;
- a frame timer;
- a polling timer;
- frame callbacks;
- registered `PolledObservable`s;
- resource cleanup registrations;
- child `PanelHandle`s.

A panel object is just a description. A `PanelHandle` is the mounted runtime
object. Handles store the panel's layout cell, children, registered resources,
and arbitrary data.

The normal pattern is:

```julia
using InteractiveIsing.Windows
using GLMakie

struct MyPanel <: AbstractPanel
    graph
end

function Windows.mount!(panel::MyPanel, host::WindowHost, cell; kwargs...)
    grid = GridLayout(cell)
    handle = PanelHandle(panel, host, grid)

    value = Observable(0)
    Label(grid[1, 1], lift(string, value))

    register_frame!(handle) do _
        value[] += 1
        return nothing
    end

    return handle
end
```

Mount it with:

```julia
host = window()
panel!(host, :counter, MyPanel(g), (1, 1))
```

## Resource Ownership

Panels should register anything that needs cleanup:

```julia
register!(handle, on(obs) do x
    # callback
end)

register_frame!(handle) do host
    # frame callback
end

register_polled!(handle, PolledObservable(initial, _ -> compute_value()))
```

When a panel closes, children close first, then the panel-specific `close!`
hook, then registered resources. Host close stops the frame and polling timers
before closing children, which avoids callbacks firing while Makie objects are
being torn down.

## Lifecycle Hooks

Panels can implement:

```julia
Windows.close!(panel::MyPanel, handle::PanelHandle) = nothing
Windows.pause!(panel::MyPanel, handle::PanelHandle) = nothing
Windows.resume!(panel::MyPanel, handle::PanelHandle) = nothing
Windows.restart!(panel::MyPanel, handle::PanelHandle) = nothing
```

Use these for panel-owned processes or state that is not already registered as a
resource.

Host pause does not pause the window frame or polling timers. It only propagates
the panel pause lifecycle. Simulation process pausing is handled by the
simulation status panel, not by stopping the UI.

## Hamiltonian Visualization Extension

Hamiltonian visualization is intentionally separate from the panel internals.
The selector asks:

```julia
hamiltonian_visualizations(term, g)
```

The fallback calls:

```julia
displayable_hamiltonian_parameters(term, g)
```

and converts the returned names into `parameter_display` specs. The default
`displayable_hamiltonian_parameters` returns all graph-sized vector parameters
from `parameters(term)`, so parameter-template Hamiltonians get a working
display with no custom UI code.

Use three levels of customization:

1. Do nothing: all state-sized parameters appear automatically.
2. Override `displayable_hamiltonian_parameters(term, g)` to filter or order
   ordinary parameters.
3. Override `hamiltonian_visualizations(term, g)` to return custom
   `HamiltonianDisplaySpec`s.

Custom layer-shaped data should use `layer_display`:

```julia
function Windows.hamiltonian_visualizations(term::MyTerm, g)
    return [
        layer_display(:derived, layer -> derived_array(term, layer);
            colorrange = :data),
    ]
end
```

Graph-sized parameter data should use `parameter_display`:

```julia
Windows.hamiltonian_visualizations(term::MyTerm, g) = [
    parameter_display(term, :field, g),
]
```

## Layer Dispatch

Layer-dependent code should use the package's inline layer dispatch helper
instead of manually branching on layer dimensionality. The Windows utilities use
`_with_layer(f, g, layer_idx)` internally, which delegates to
`inline_layer_dispatch`.

Specialize rendering methods on the layer type:

```julia
draw!(handle, layer::AbstractIsingLayer{T,2}) where T = ...
draw!(handle, layer::AbstractIsingLayer{T,3}) where T = ...
```

and call them through `_with_layer`.
