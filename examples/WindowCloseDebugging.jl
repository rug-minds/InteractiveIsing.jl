using InteractiveIsing
using InteractiveIsing.Windows

# Close-debug helper.
#
# Run this file, then call:
#
#     describe_debug_windows()
#     open_next!()
#
# Close the native window with the red dot. Then call `open_next!()` again.
# Tell Codex which names freeze and which names close cleanly.

function isingweights(; dr)
    return dr == 1 ? 1f0 : 0f0
end

wg = @WG isingweights NN = 1

debug_graph = IsingGraph(
    Layer(48, 48, Continuous(), wg),
    Layer(48, 48, Continuous(), wg),
    Ising(b = [1f0 for _ in 1:(48 * 48 * 2)]) + Quartic();
)

# For the first close-debug pass, keep the graph process-free. After the
# window-only variants are tested, call `start_debug_process!()` explicitly if
# you want to repeat the sequence with a running algorithm.
function start_debug_process!(; algorithm = InteractiveIsing.LocalLangevin(adjusted = false), kwargs...)
    return createProcess(debug_graph, algorithm; kwargs...)
end

const CLOSE_DEBUG_NAMES = close_debug_window_names()
const CLOSE_DEBUG_INDEX = Ref(0)
const CLOSE_DEBUG_LAST = Ref{Any}(nothing)
const CLOSE_DEBUG_PROCESS_NAMES = [
    :layer_view_running_langevin,
    :status_running_langevin,
    :simulation_running_langevin,
    :public_interface_running_langevin,
    :simulation_open_then_langevin,
    :public_interface_open_then_langevin,
    :simulation_open_then_metropolis,
    :public_interface_open_then_metropolis,
    :subset_status_layer_langevin,
    :subset_layer_only_langevin,
    :subset_layer_only_empty_on_close_langevin,
    :subset_layer_only_copy_langevin,
    :subset_layer_only_no_polltimer_langevin,
    :subset_temperature_only_langevin,
    :subset_kinetic_only_langevin,
    :subset_layer_temperature_no_kinetic_langevin,
    :subset_layer_kinetic_langevin,
    :subset_layer_temperature_langevin,
    :subset_layer_hamiltonian_langevin,
    :subset_layer_magnetization_langevin,
    :subset_status_layer_temperature_langevin,
    :subset_full_no_status_langevin,
    :subset_full_no_temperature_langevin,
    :subset_full_no_hamiltonian_langevin,
    :subset_full_no_magnetization_langevin,
]

const CLOSE_DEBUG_SUSPECT_NAMES = [
    :subset_layer_only_langevin,
    :subset_layer_only_empty_on_close_langevin,
    :subset_layer_only_copy_langevin,
    :subset_layer_only_no_polltimer_langevin,
    :subset_temperature_only_langevin,
    :subset_kinetic_only_langevin,
    :subset_layer_temperature_no_kinetic_langevin,
    :subset_layer_kinetic_langevin,
    :subset_layer_temperature_langevin,
]
const CLOSE_DEBUG_SUSPECT_INDEX = Ref(0)

const CLOSE_DEBUG_SCENARIOS = [
    :example_3d_langevin_coulomb,
    :example_3d_coulomb,
    :example_3d_default,
    :example_3d_kinetic,
    :example_global_langevin_unbounded,
    :example_2d_two_layer_langevin,
]

function describe_debug_windows()
    for (i, spec) in enumerate(close_debug_window_descriptions())
        println(lpad(i, 2), ". ", rpad(String(spec.name), 40), spec.description)
    end
    return nothing
end

function open_debug!(name::Symbol; graph = debug_graph, kwargs...)
    println("Opening close-debug window: ", name)
    CLOSE_DEBUG_LAST[] = open_close_debug_window(graph, name; kwargs...)
    return nothing
end

function open_next!(; graph = debug_graph, kwargs...)
    CLOSE_DEBUG_INDEX[] += 1
    if CLOSE_DEBUG_INDEX[] > length(CLOSE_DEBUG_NAMES)
        println("No more close-debug windows.")
        return nothing
    end
    return open_debug!(CLOSE_DEBUG_NAMES[CLOSE_DEBUG_INDEX[]]; graph, kwargs...)
end

function reset_debug_sequence!()
    CLOSE_DEBUG_INDEX[] = 0
    return nothing
end

function reset_debug_graph!()
    try
        close(debug_graph)
    catch err
        @warn "Could not close old debug graph processes" exception = (err, catch_backtrace())
    end
    return debug_graph
end

"""
    open_process_debug!(name = :simulation_running_langevin; process_algorithm = LocalLangevin(...), kwargs...)

Open a debug window that starts a graph process. Pass
`process_algorithm = nothing` to use the package default process constructor, or
pass another algorithm to check whether the close behavior depends on the exact
process implementation.
"""
function open_process_debug!(name::Symbol = :simulation_running_langevin; kwargs...)
    name in CLOSE_DEBUG_PROCESS_NAMES || @warn "Opening a non-process debug window through open_process_debug!" name
    reset_debug_graph!()
    return open_debug!(name; graph = debug_graph, kwargs...)
end

"""
    open_suspect_debug!(name; process_algorithm = LocalLangevin(...), kwargs...)

Open one focused suspect window. The suspect names still contain `langevin` for
history, but callers can override `process_algorithm`; the intended diagnostic
is a running graph process, not Langevin specifically.
"""
function open_suspect_debug!(name::Symbol; kwargs...)
    name in CLOSE_DEBUG_SUSPECT_NAMES || throw(ArgumentError("Unknown suspect close-debug window $name. Use CLOSE_DEBUG_SUSPECT_NAMES."))
    return open_process_debug!(name; kwargs...)
end

function open_next_suspect!(; kwargs...)
    CLOSE_DEBUG_SUSPECT_INDEX[] += 1
    if CLOSE_DEBUG_SUSPECT_INDEX[] > length(CLOSE_DEBUG_SUSPECT_NAMES)
        println("No more suspect close-debug windows.")
        return nothing
    end
    return open_suspect_debug!(CLOSE_DEBUG_SUSPECT_NAMES[CLOSE_DEBUG_SUSPECT_INDEX[]]; kwargs...)
end

function reset_suspect_sequence!()
    CLOSE_DEBUG_SUSPECT_INDEX[] = 0
    return nothing
end

function _scenario_nn_weight(; dr)
    return dr == 1 ? 1f0 : 0f0
end

function _scenario_inverse_weight(dr, c1, c2)
    return 1 / dr
end

function _scenario_ising_weight(dr, c1, c2)
    return dr == 1 ? 1.0f0 : 0.0f0
end

function make_close_debug_scenario(name::Symbol)
    if name === :example_3d_langevin_coulomb
        wg3 = @WG _scenario_nn_weight NN = 1
        g = IsingGraph(
            40,
            40,
            10,
            Continuous(),
            wg3,
            LatticeConstants(1f0, 1f0, 1f0),
            StateSet(-1.5f0, 1.5f0),
            Ising(c = ConstVal(0f0), b = 0) + CoulombHamiltonian(recalc = 5000),
            periodic = (:x, :y),
        )
        algorithm = LocalLangevin(stepsize = 0.1f0, adjusted = true)
        return (; graph = g, algorithm, title = "Scenario: 3D Langevin Coulomb")
    elseif name === :example_3d_coulomb
        wg3 = @WG _scenario_nn_weight NN = 1
        g = IsingGraph(
            40,
            40,
            10,
            Continuous(),
            wg3,
            LatticeConstants(1f0, 1f0, 1f0),
            StateSet(-1.5f0, 1.5f0),
            Ising(c = ConstVal(0f0), b = 0) + CoulombHamiltonian(scaling = 1f0, screening = Inf32, recalc = 200),
            periodic = (:x, :y),
            precision = Float32,
        )
        temp!(g, 1f0)
        algorithm = LocalLangevin(stepsize = 0.1f0, adjusted = true)
        return (; graph = g, algorithm, title = "Scenario: 3D Coulomb")
    elseif name === :example_3d_default
        wg3 = @WG _scenario_nn_weight NN = 1
        g = IsingGraph(
            100,
            100,
            10,
            Continuous(),
            LocalProposer(0.5),
            wg3,
            LatticeConstants(1f0, 1f0, 1f0),
            StateSet(-1f0, 1f0),
            Ising(c = ConstVal(0f0), b = 0, localpotential = 0),
            periodic = (:x, :y),
        )
        return (; graph = g, algorithm = nothing, title = "Scenario: 3D Default")
    elseif name === :example_3d_kinetic
        wg3 = @WG (dr, c1, c2) -> _scenario_inverse_weight(dr, c1, c2) NN = (3, 3, 2)
        g = IsingGraph(
            100,
            100,
            10,
            Continuous(),
            wg3,
            LatticeConstants(1.0, 1.0, 20.0),
            StateSet(-1.5f0, 1.5f0),
            Ising(b = :homogeneous) + Clamping(1f0) + Quartic() + Sextic(),
            periodic = (:x, :y),
            self = :homogeneous,
        )
        return (; graph = g, algorithm = KineticMC(), title = "Scenario: 3D Kinetic")
    elseif name === :example_global_langevin_unbounded
        wg3 = @WG _scenario_nn_weight NN = 1
        g = IsingGraph(
            30,
            30,
            10,
            Continuous(),
            wg3,
            LatticeConstants(1f0, 1f0, 1f0),
            StateSet(-Inf32, Inf32),
            Ising(c = ConstVal(16f0), localpotential = ConstFill(1f0), b = ConstFill(4f0)),
            periodic = (:x, :y),
        )
        temp!(g, 18.64f0)
        state(g) .= 0f0
        algorithm = GlobalLangevin(stepsize = 0.001f0, adjusted = true)
        return (; graph = g, algorithm, title = "Scenario: Global Langevin Unbounded")
    elseif name === :example_2d_two_layer_langevin
        wg2 = @WG _scenario_nn_weight NN = 1
        g = IsingGraph(
            Layer(64, 64, Continuous(), wg2),
            Layer(64, 64, Continuous(), wg2),
            Ising(b = [1f0 for _ in 1:(64 * 64 * 2)]) + Quartic();
        )
        algorithm = LocalLangevin(stepsize = 0.1f0, adjusted = true)
        return (; graph = g, algorithm, title = "Scenario: 2D Two-Layer Langevin")
    end
    throw(ArgumentError("Unknown close-debug scenario $name. Use CLOSE_DEBUG_SCENARIOS."))
end

function open_scenario_debug!(name::Symbol = :example_3d_langevin_coulomb; start_process = true, kwargs...)
    name in CLOSE_DEBUG_SCENARIOS || throw(ArgumentError("Unknown close-debug scenario $name. Use CLOSE_DEBUG_SCENARIOS."))
    scenario = make_close_debug_scenario(name)
    println("Opening close-debug scenario: ", name)
    host = interface(scenario.graph; title = scenario.title, kwargs...)
    if start_process
        if isnothing(scenario.algorithm)
            createProcess(scenario.graph)
        else
            createProcess(scenario.graph, scenario.algorithm)
        end
    end
    CLOSE_DEBUG_LAST[] = host
    return nothing
end

describe_debug_windows()
println()
println("Call open_next!(), close that window with the red dot, then call open_next!() again.")
println("For the process path, call open_process_debug!(), or pass one of:")
println(CLOSE_DEBUG_PROCESS_NAMES)
println("For the current suspected neighborhood, call open_next_suspect!(), or pass one of:")
println(CLOSE_DEBUG_SUSPECT_NAMES)
println("For exact example-like scenarios, call open_scenario_debug!(), or pass one of:")
println(CLOSE_DEBUG_SCENARIOS)
