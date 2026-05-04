"""
    InteractiveIsing.Windows

Experimental GLMakie window platform for interactive Ising graph interfaces.

The module provides a small host/panel framework:

- `window` creates a `WindowHost`, which owns the Makie figure,
  screen, frame timer, polling timer, resources, and mounted panels.
- `panel!` mounts an `AbstractPanel` into a host or parent
  `PanelHandle`.
- `close`, `pause!`, `resume!`, and `restart!`
  propagate lifecycle operations through the panel tree.
- `interface` opens the default simulation UI.

Panel authors usually define a concrete subtype of `AbstractPanel` and
extend `mount!`.
"""
module Windows

using DataStructures
using GLMakie
using GLMakie.GLFW
using GLMakie: to_native
using Observables
using Preferences
using SparseArrays
using UUIDs

using ..Processes
using ..InteractiveIsing: AbstractIsingLayer, Clamping, CoulombHamiltonian,
    CastVec, HamiltonianTerms, IsingGraph, MagField, PTimer, PolynomialHamiltonian,
    PolledObservable, Parameter, SingleLayerGraph, addRandomDefects!, adj, graphidxs,
    hamiltonian, hamiltonians, idxToCoord, inline_layer_dispatch, layers,
    modulefolder, nstates, origin, parameters, plotCorr, poll!, processes,
    saveGImg, state, stateset, temp, temp!, value, wg

export AbstractPanel, PanelHandle, WindowHost, SimulationPanel, StatusPanel,
    ConnectionsPanel, ContextLinesPanel, HamiltonianDisplaySpec, HamiltonianParameterPanel,
    LayerDisplayValue, LayerSelectorPanel, LayerViewPanel, TemperaturePanel, MagnetizationPanel,
    close!, interface, layer_display, mount!, new_interface, panel!,
    parameter_display, pause!, register!, register_frame!, register_polled!,
    restart!, resume!, window, displayable_hamiltonian_parameters,
    hamiltonian_visualizations

include("Core.jl")
include("Panels/Panels.jl")

"""
    interface(g; kwargs...) -> WindowHost

Open the Windows-based default simulation interface for graph `g`. Keyword
arguments are forwarded to `new_interface`.
"""
interface(g; kwargs...) = new_interface(g; kwargs...)

end

using .Windows: interface, new_interface
export Windows, interface, new_interface
