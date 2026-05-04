module Windows

using DataStructures
using GLMakie
using GLMakie.GLFW
using GLMakie: to_native
using Observables
using Preferences
using UUIDs

using ..Processes
using ..InteractiveIsing: AbstractIsingLayer, AverageCircular, Clamping, CoulombHamiltonian,
    CastVec, HamiltonianTerms, IsingGraph, MagField, PTimer, PolynomialHamiltonian,
    PolledObservable, Parameter, SingleLayerGraph, addRandomDefects!, graphidxs, hamiltonian,
    hamiltonians, inline_layer_dispatch, layers, avg,
    modulefolder, nstates, origin, parameters, plotCorr, poll!, processes,
    saveGImg, state, stateset, temp, temp!, value, wg

export AbstractPanel, PanelHandle, WindowHost, SimulationPanel, StatusPanel,
    HamiltonianDisplaySpec, HamiltonianParameterPanel, LayerDisplayValue,
    LayerSelectorPanel, LayerViewPanel, TemperaturePanel, MagnetizationPanel,
    close!, interface, layer_display, mount!, new_interface, panel!,
    parameter_display, pause!, register!, register_frame!, register_polled!,
    restart!, resume!, window, displayable_hamiltonian_parameters,
    hamiltonian_visualizations

include("Core.jl")
include("Panels/Panels.jl")

interface(g; kwargs...) = new_interface(g; kwargs...)

end

using .Windows: interface, new_interface
export Windows, interface, new_interface
