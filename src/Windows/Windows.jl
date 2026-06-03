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
using LinearAlgebra
using SparseArrays
using UUIDs

using ..Processes
using ..InteractiveIsing: AbstractIsingLayer, AbstractLayerTopology, AbstractSpinGraph, AbstractVectorSpinGraph,
    Clamping, CoulombHamiltonian, CastVec, HamiltonianTerms, InteractiveGraphVarSpec, IsingGraph, KineticMC, MagField, PTimer, PolynomialHamiltonian,
    PolledObservable, Parameter, SingleLayerGraph, UndirectedAdjacency, _mc_model_inits, _prepared_interactive_var_data,
    _resolve_interactive_target_key, _set_interactive_graph_var_value!, addRandomDefects!, adj, createProcess, graph, graphidxs,
    Coordinate, hamiltonian, hamiltonians, idxToCoord, inline_layer_dispatch, interactivevars, layers, modulefolder,
    nstates, origin, parameters, plotCorr, poll!, processes, coords, reset!, saveGImg, spin_dimension, state, VectorExchange,
    stateset, temp, temp!, topology, value, wg, woorldcoordinate

export AbstractPanel, PanelHandle, WindowHost, SimulationPanel, StatusPanel,
    AllLayersViewPanel, ConnectionsPanel, ContextLinesPanel, HamiltonianDisplaySpec, HamiltonianParameterPanel,
    InteractiveLinesPanel, InteractiveVariablesPanel, KineticTimePanel, LayerDisplayValue, LayerSelectorPanel, LayerViewPanel,
    TemperaturePanel, MagnetizationPanel,
    HasAxis, NoAxis, AxisTrait, HasImage, NoImage, ImageTrait,
    axis_to_png, axis_trait, axiskey, fullimage, image_trait,
    close!, detach_hot_observable!, getaxis, hasaxis, hasimage, hot_observable!,
    hot_observable_zero,
    interface, layer_display, mount!, new_interface, onclose!, panel!,
    parameter_display, pause!, register!, register_frame!, register_hot_observable!,
    register_polled!, restart!, resume!, tofigure, toimage, toimage!,
    topology_layer_display!, window,
    displayable_hamiltonian_parameters, hamiltonian_visualizations,
    close_debug_window_descriptions, close_debug_window_names,
    open_close_debug_window

include("Core.jl")
include("Interfaces.jl")
include("Panels/Panels.jl")
include("Debugging/CloseDebugWindows.jl")

"""
    interface(g; kwargs...) -> WindowHost

Open the Windows-based default simulation interface for graph `g`. Keyword
arguments are forwarded to `new_interface`.
"""
interface(g; kwargs...) = new_interface(g; kwargs...)

end

using .Windows: interface, new_interface
export Windows, interface, new_interface
