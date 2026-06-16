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
using Unitful
using UUIDs

using ..StatefulAlgorithms
using ..InteractiveIsing: AbstractIsingLayer, AbstractLayerTopology, Clamping, CoulombHamiltonian,
    CastVec, HamiltonianTerms, InteractiveGraphVarSpec, IsingGraph, KineticMC, MagField, PTimer, PolynomialHamiltonian,
    PolledObservable, Parameter, SingleLayerGraph, _mc_model_inits, _prepared_interactive_var_data,
    _resolve_interactive_target_key, _set_interactive_graph_var_value!, addRandomDefects!, adj, createProcess, graphidxs,
    Coordinate, hamiltonian, hamiltonians, idxToCoord, inline_layer_dispatch, interactivevars, layers, modulefolder,
    nstates, origin, parameters, plotCorr, poll!, processes, coords, reset!, saveGImg, state,
    stateset, temp, temp!, settemp!, topology, value, wg, woorldcoordinate, physicalscales

export AbstractPanel, PanelHandle, WindowHost, SimulationPanel, StatusPanel,
    AllLayersViewPanel, ConnectionsPanel, ContextLinesPanel, HamiltonianDisplaySpec, HamiltonianParameterPanel,
    InteractiveLinesPanel, InteractiveVariablesPanel, KineticTimePanel, LayerDisplayValue, LayerSelectorPanel, LayerViewPanel,
    TemperaturePanel, MagnetizationPanel,
    HasAxis, NoAxis, AxisTrait, HasImage, NoImage, ImageTrait,
    axis_to_png, axis_trait, axiskey, fullimage, image_trait,
    close!, detach_hot_observable!, getaxis, hasaxis, hasimage, hot_observable!,
    hot_observable_zero,
    fill_topology_layer_axis!, interface, layer_display, mount!, new_interface, onclose!, panel!,
    parameter_display, pause!, register!, register_frame!, register_hot_observable!,
    register_polled!, restart!, resume!, run_interface, tofigure, toimage, toimage!,
    window,
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

"""
    run_interface(g, func = nothing, inputs...; dynamics = g.default_algorithm, interactive = nothing, interface_kwargs = (;), kwargs...)

Open the default graph interface and start a graph process. Process-facing
keywords are forwarded to `createProcess`. UI-facing keywords can be passed
directly for the standard interface options, or collected in `interface_kwargs`.

By default this enables the standard graph-interactive temperature behavior
before creating the process, so the process `:T` variable is interactive. Pass
`interactive = false` to leave the graph addon unchanged, or an explicit
`Interactive(...)`/tuple of `Interactive(...)` specs to choose variables.
"""
function run_interface(
    g::IsingGraph,
    func = nothing,
    inputs...;
    dynamics = g.default_algorithm,
    interactive = nothing,
    interface_kwargs = (;),
    framerate = nothing,
    polling_rate = nothing,
    size = nothing,
    title = nothing,
    hide_left_buttons = nothing,
    kwargs...,
)
    # Store graph-level interactivity before process creation so createProcess
    # can resolve it against the prepared algorithm context.
    if interactive === true
        g.addons[:interactive] = true
    elseif interactive === false
        nothing
    elseif isnothing(interactive)
        g.addons[:interactive] = true
    else
        g.addons[:interactive] = interactive
    end

    ui_kwargs = (; interface_kwargs...)
    isnothing(framerate) || (ui_kwargs = (; ui_kwargs..., framerate))
    isnothing(polling_rate) || (ui_kwargs = (; ui_kwargs..., polling_rate))
    isnothing(size) || (ui_kwargs = (; ui_kwargs..., size))
    isnothing(title) || (ui_kwargs = (; ui_kwargs..., title))
    isnothing(hide_left_buttons) || (ui_kwargs = (; ui_kwargs..., hide_left_buttons))

    host = interface(g; ui_kwargs...)
    process = createProcess(g, func, inputs...; dynamics, kwargs...)
    return (; host, process)
end

end

using .Windows: interface, new_interface, run_interface
export Windows, interface, new_interface, run_interface
