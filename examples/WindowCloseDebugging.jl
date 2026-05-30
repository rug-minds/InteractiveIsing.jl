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
    :all_layers_view_running_langevin,
    :status_running_langevin,
    :simulation_running_langevin,
    :public_interface_running_langevin,
    :simulation_open_then_langevin,
    :public_interface_open_then_langevin,
    :simulation_open_then_metropolis,
    :public_interface_open_then_metropolis,
    :subset_status_layer_langevin,
    :subset_layer_only_langevin,
    :all_layers_view_open_then_langevin,
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
    :all_layers_view_open_then_langevin,
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

const CLOSE_DEBUG_HOT_SUSPECT_NAMES = [
    :subset_layer_only_langevin,
    :all_layers_view_open_then_langevin,
    :subset_layer_hamiltonian_langevin,
    :subset_full_no_hamiltonian_langevin,
    :subset_full_no_temperature_langevin,
    :public_interface_open_then_langevin,
]
const CLOSE_DEBUG_HOT_SUSPECT_INDEX = Ref(0)

const CLOSE_DEBUG_INTERFACE_SUSPECT_NAMES = [
    :public_interface_open_then_langevin,
    :public_interface_running_langevin,
    :simulation_open_then_langevin,
    :simulation_running_langevin,
    :public_interface_open_then_metropolis,
    :simulation_open_then_metropolis,
]
const CLOSE_DEBUG_INTERFACE_SUSPECT_INDEX = Ref(0)
const CLOSE_DEBUG_REPEAT_NAME = Ref(:public_interface_open_then_langevin)
const CLOSE_DEBUG_REPEAT_LEFT = Ref(0)
const CLOSE_DEBUG_REPEAT_TOTAL = Ref(0)
const CLOSE_DEBUG_EXACT_GRAPH = Ref{Any}(nothing)
const CLOSE_DEBUG_EXACT_ALGORITHM = Ref{Any}(InteractiveIsing.LocalLangevin(adjusted = false))
const CLOSE_DEBUG_EXACT_ENTRY = Ref(:interface)
const CLOSE_DEBUG_EXACT_START_PROCESS = Ref(true)
const CLOSE_DEBUG_EXACT_PROCESS_KWARGS = Ref{Any}(NamedTuple())
const CLOSE_DEBUG_EXACT_REPEAT_LEFT = Ref(0)
const CLOSE_DEBUG_EXACT_REPEAT_TOTAL = Ref(0)
const CLOSE_DEBUG_SCENARIO_REPEAT_NAME = Ref(:example_3d_langevin_coulomb)
const CLOSE_DEBUG_SCENARIO_REPEAT_START_PROCESS = Ref(true)
const CLOSE_DEBUG_SCENARIO_REPEAT_LEFT = Ref(0)
const CLOSE_DEBUG_SCENARIO_REPEAT_TOTAL = Ref(0)

const CLOSE_DEBUG_SCENARIOS = [
    :example_hexagonal,
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

function open_debug!(name::Symbol; graph = debug_graph, trace = false, kwargs...)
    println("Opening close-debug window: ", name)
    CLOSE_DEBUG_LAST[] = open_close_debug_window(graph, name; kwargs...)
    if trace !== false
        trace_last_close!(trace)
    end
    return nothing
end

"""
    trace_last_close!(enabled = true)

Enable close-phase tracing on the most recently opened debug window. The trace
prints phase durations when the window is closed, which is useful for slow macOS
closes that eventually release.
"""
function trace_last_close!(enabled = true)
    host = CLOSE_DEBUG_LAST[]
    isnothing(host) && return println("No close-debug host has been opened.")
    host[:close_trace] = enabled
    println(enabled === false ? "Close tracing disabled." : "Close tracing enabled.")
    return host
end

"""
    close_trace_log(host = CLOSE_DEBUG_LAST[])

Return the raw close trace log for the last debug window.
"""
function close_trace_log(host = CLOSE_DEBUG_LAST[])
    isnothing(host) && return Any[]
    return get(host.data, :close_trace_log, Any[])
end

"""
    print_close_trace(host = CLOSE_DEBUG_LAST[])

Print the close trace with per-phase durations after a slow close eventually
releases.
"""
function print_close_trace(host = CLOSE_DEBUG_LAST[])
    log = close_trace_log(host)
    isempty(log) && return println("No close trace has been recorded.")
    previous_time = first(log).time
    for row in log
        println(
            rpad(String(row.phase), 34),
            " dt=",
            round(row.time - previous_time; digits = 4),
            " hot=",
            row.hot_observables,
            " method=",
            get(row, :method, :unknown),
            " closing=",
            row.closing,
            " closed=",
            row.closed,
        )
        previous_time = row.time
    end
    return log
end

"""
    mark_manual_close!(method = :native, host = CLOSE_DEBUG_LAST[])

Label the next manual close attempt in the trace. Use `:native` before clicking
the red dot and `:keyboard` before pressing Cmd+W if the key handler itself is
not firing for a particular platform/backend path.
"""
function mark_manual_close!(method::Symbol = :native, host = CLOSE_DEBUG_LAST[])
    isnothing(host) && return println("No close-debug host has been opened.")
    host[:close_method] = method
    println("Marked next close method as ", method)
    return host
end

"""
    close_last_programmatically!(host = CLOSE_DEBUG_LAST[])

Close the most recently opened debug window through `close(host)`. This tests
the explicit API path separately from the native red-dot and Cmd+W paths.
"""
function close_last_programmatically!(host = CLOSE_DEBUG_LAST[])
    isnothing(host) && return println("No close-debug host has been opened.")
    host[:close_trace] = get(host.data, :close_trace, true)
    close(host)
    return host
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

"""
    open_next_hot_suspect!(; kwargs...)

Open the next short-list close suspect focused on hot/dynamic plot observables.
This avoids walking the full debug matrix when checking close freezes on macOS.
"""
function open_next_hot_suspect!(; kwargs...)
    CLOSE_DEBUG_HOT_SUSPECT_INDEX[] += 1
    if CLOSE_DEBUG_HOT_SUSPECT_INDEX[] > length(CLOSE_DEBUG_HOT_SUSPECT_NAMES)
        println("No more hot-observable suspect close-debug windows.")
        return nothing
    end
    return open_process_debug!(CLOSE_DEBUG_HOT_SUSPECT_NAMES[CLOSE_DEBUG_HOT_SUSPECT_INDEX[]]; kwargs...)
end

"""
    reset_hot_suspect_sequence!()

Reset the short hot-observable suspect sequence used by
`open_next_hot_suspect!`.
"""
function reset_hot_suspect_sequence!()
    CLOSE_DEBUG_HOT_SUSPECT_INDEX[] = 0
    return nothing
end

"""
    open_next_interface_suspect!(; trace = true, kwargs...)

Open the next normal-interface close suspect. This sequence is restricted to
`interface(g)` and `SimulationPanel` paths with ordinary graph processes, so it
is the right pass when plain `interface(g)` freezes.
"""
function open_next_interface_suspect!(; trace = true, kwargs...)
    CLOSE_DEBUG_INTERFACE_SUSPECT_INDEX[] += 1
    if CLOSE_DEBUG_INTERFACE_SUSPECT_INDEX[] > length(CLOSE_DEBUG_INTERFACE_SUSPECT_NAMES)
        println("No more interface close-debug suspects.")
        return nothing
    end
    return open_process_debug!(CLOSE_DEBUG_INTERFACE_SUSPECT_NAMES[CLOSE_DEBUG_INTERFACE_SUSPECT_INDEX[]]; trace, kwargs...)
end

"""
    reset_interface_suspect_sequence!()

Reset the normal-interface suspect sequence used by
`open_next_interface_suspect!`.
"""
function reset_interface_suspect_sequence!()
    CLOSE_DEBUG_INTERFACE_SUSPECT_INDEX[] = 0
    return nothing
end

"""
    repeat_debug!(name = :public_interface_open_then_langevin; repeats = 10)

Prepare a repeated close-debug run for one suspect. Use `open_next_repeat!()`,
close the native window, then call `open_next_repeat!()` again until the helper
reports that the run is complete.
"""
function repeat_debug!(name::Symbol = :public_interface_open_then_langevin; repeats = 10)
    name in CLOSE_DEBUG_PROCESS_NAMES || throw(ArgumentError("Unknown process debug window $name."))
    CLOSE_DEBUG_REPEAT_NAME[] = name
    CLOSE_DEBUG_REPEAT_LEFT[] = Int(repeats)
    CLOSE_DEBUG_REPEAT_TOTAL[] = Int(repeats)
    println("Prepared repeated close-debug run for ", name, " x ", repeats)
    return nothing
end

"""
    open_next_repeat!(; trace = true, kwargs...)

Open the next window in the repeated suspect run prepared by `repeat_debug!`.
Tracing is enabled by default because the freeze is intermittent and the slow
phase matters when a close eventually releases.
"""
function open_next_repeat!(; trace = true, kwargs...)
    if CLOSE_DEBUG_REPEAT_LEFT[] <= 0
        println("Repeated close-debug run complete.")
        return nothing
    end
    iteration = CLOSE_DEBUG_REPEAT_TOTAL[] - CLOSE_DEBUG_REPEAT_LEFT[] + 1
    CLOSE_DEBUG_REPEAT_LEFT[] -= 1
    println(
        "Repeat close-debug ",
        iteration,
        "/",
        CLOSE_DEBUG_REPEAT_TOTAL[],
        ": ",
        CLOSE_DEBUG_REPEAT_NAME[],
    )
    return open_process_debug!(CLOSE_DEBUG_REPEAT_NAME[]; trace, kwargs...)
end

"""
    open_exact_interface_debug!(g; entry = :interface, start_process = true,
                                process_algorithm = LocalLangevin(...),
                                process_kwargs = NamedTuple(), trace = true,
                                kwargs...)

Open a close-debug window using the exact graph object `g` instead of the
synthetic `debug_graph`. This is the preferred path when the generic suspects
do not reproduce a freeze that happens in a real `interface(g)` run.

`entry = :interface` calls the public `interface(g)`. `entry = :simulation`
opens the same `SimulationPanel` through the debug host. When `start_process`
is true, the process starts after the window is displayed, matching the common
manual workflow.
"""
function open_exact_interface_debug!(
    g;
    entry = :interface,
    start_process = true,
    process_algorithm = InteractiveIsing.LocalLangevin(adjusted = false),
    process_kwargs = NamedTuple(),
    trace = true,
    kwargs...,
)
    name =
        entry === :interface ? :public_interface :
        entry === :simulation ? :simulation_full :
        throw(ArgumentError("entry must be :interface or :simulation, got $entry"))

    println("Opening exact close-debug window: ", name)
    CLOSE_DEBUG_LAST[] = open_close_debug_window(g, name; kwargs...)
    if trace !== false
        trace_last_close!(trace)
    end
    if start_process
        _start_exact_debug_process!(g, process_algorithm; process_kwargs...)
    end
    return CLOSE_DEBUG_LAST[]
end

"""
    repeat_exact_interface_debug!(g; repeats = 10, kwargs...)

Prepare a repeated close-debug run using the exact graph `g`. Call
`open_next_exact_repeat!()`, close the native window, then repeat.
"""
function repeat_exact_interface_debug!(
    g;
    repeats = 10,
    entry = :interface,
    start_process = true,
    process_algorithm = InteractiveIsing.LocalLangevin(adjusted = false),
    process_kwargs = NamedTuple(),
)
    CLOSE_DEBUG_EXACT_GRAPH[] = g
    CLOSE_DEBUG_EXACT_ALGORITHM[] = process_algorithm
    CLOSE_DEBUG_EXACT_ENTRY[] = Symbol(entry)
    CLOSE_DEBUG_EXACT_START_PROCESS[] = Bool(start_process)
    CLOSE_DEBUG_EXACT_PROCESS_KWARGS[] = process_kwargs
    CLOSE_DEBUG_EXACT_REPEAT_LEFT[] = Int(repeats)
    CLOSE_DEBUG_EXACT_REPEAT_TOTAL[] = Int(repeats)
    println("Prepared exact graph close-debug run x ", repeats, " with entry ", entry)
    return nothing
end

"""
    open_next_exact_repeat!(; trace = true, kwargs...)

Open the next exact-graph debug window prepared by
`repeat_exact_interface_debug!`.
"""
function open_next_exact_repeat!(; trace = true, kwargs...)
    isnothing(CLOSE_DEBUG_EXACT_GRAPH[]) && throw(ArgumentError("Call repeat_exact_interface_debug!(g) first."))
    if CLOSE_DEBUG_EXACT_REPEAT_LEFT[] <= 0
        println("Exact graph close-debug run complete.")
        return nothing
    end
    iteration = CLOSE_DEBUG_EXACT_REPEAT_TOTAL[] - CLOSE_DEBUG_EXACT_REPEAT_LEFT[] + 1
    CLOSE_DEBUG_EXACT_REPEAT_LEFT[] -= 1
    println(
        "Exact graph close-debug ",
        iteration,
        "/",
        CLOSE_DEBUG_EXACT_REPEAT_TOTAL[],
        " entry=",
        CLOSE_DEBUG_EXACT_ENTRY[],
    )
    return open_exact_interface_debug!(
        CLOSE_DEBUG_EXACT_GRAPH[];
        entry = CLOSE_DEBUG_EXACT_ENTRY[],
        start_process = CLOSE_DEBUG_EXACT_START_PROCESS[],
        process_algorithm = CLOSE_DEBUG_EXACT_ALGORITHM[],
        process_kwargs = CLOSE_DEBUG_EXACT_PROCESS_KWARGS[],
        trace,
        kwargs...,
    )
end

"""
    _start_exact_debug_process!(g, algorithm; kwargs...)

Start a graph process for exact-graph close debugging. `algorithm = nothing`
uses the package default `createProcess(g)` path.
"""
function _start_exact_debug_process!(g, algorithm; kwargs...)
    if isnothing(algorithm)
        return createProcess(g; allow_multiple = true, kwargs...)
    else
        return createProcess(g, algorithm; allow_multiple = true, kwargs...)
    end
end

"""
    hot_observable_report(host = CLOSE_DEBUG_LAST[])

Print the registered hot observables in `host`, including their current value
type, size, and length. Use this before closing a suspect window to verify what
the close path is expected to detach.
"""
function hot_observable_report(host = CLOSE_DEBUG_LAST[])
    isnothing(host) && return println("No close-debug host has been opened.")
    rows = _hot_observable_rows(host)
    println("Registered hot observables: ", length(rows))
    for (idx, row) in enumerate(rows)
        println(
            lpad(idx, 2),
            ". ",
            row.path,
            " | obs=",
            row.observable_type,
            " | value=",
            row.value_type,
            " | size=",
            row.value_size,
            " | length=",
            row.value_length,
        )
    end
    return rows
end

"""
    detach_hot_observables_now!(host = CLOSE_DEBUG_LAST[])

Run the same hot-observable detach pass used by native close scheduling, then
print the post-detach hot observable report. This is useful for checking whether
manual pre-detach changes close behavior on macOS.
"""
function detach_hot_observables_now!(host = CLOSE_DEBUG_LAST[])
    isnothing(host) && return println("No close-debug host has been opened.")
    InteractiveIsing.Windows._detach_hot_observables!(host)
    return hot_observable_report(host)
end

"""
    _hot_observable_rows(host_or_handle, path = "host")

Collect display rows for registered `Windows.HotObservable` resources in a host
or panel subtree.
"""
function _hot_observable_rows(owner, path = "host")
    rows = NamedTuple[]
    for resource in owner.resources
        resource isa InteractiveIsing.Windows.HotObservable || continue
        observable = resource.observable
        value = observable[]
        push!(
            rows,
            (;
                path,
                observable_type = typeof(observable),
                value_type = typeof(value),
                value_size = _hot_value_size(value),
                value_length = _hot_value_length(value),
            ),
        )
    end
    for (key, child) in pairs(owner.children)
        append!(rows, _hot_observable_rows(child, string(path, " / ", key)))
    end
    return rows
end

"""
    _hot_value_size(value)

Return `size(value)` for array-like hot observable values, or `nothing` for
non-sized values.
"""
function _hot_value_size(value)
    try
        return size(value)
    catch
        return nothing
    end
end

"""
    _hot_value_length(value)

Return `length(value)` for length-aware hot observable values, or `nothing` for
non-container values.
"""
function _hot_value_length(value)
    try
        return length(value)
    catch
        return nothing
    end
end

function _scenario_nn_weight(; dr)
    return dr == 1 ? 1f0 : 0f0
end

function _scenario_zigzag_hexagonal_isingweights(; dr::R) where {R}
    return isapprox(dr, one(R); atol = R(1e-6)) ? 1f0 : 0f0
end

function _scenario_inverse_weight(dr, c1, c2)
    return 1 / dr
end

function _scenario_ising_weight(dr, c1, c2)
    return dr == 1 ? 1.0f0 : 0.0f0
end

function make_close_debug_scenario(name::Symbol)
    if name === :example_hexagonal
        row_spacing = sqrt(3f0) / 2
        top = LatticeTopology(
            (0f0, row_spacing, 0f0),
            (1f0, 0f0, 0f0),
            (0f0, 0f0, 1f0);
            layout = ZigZagRows(),
            periodic = true,
            lattice_type = Hexagonal,
        )
        wg = @WG _scenario_zigzag_hexagonal_isingweights NN = 1
        g = IsingGraph(
            40,
            40,
            10,
            Continuous(),
            LocalProposer(0.5f0),
            wg,
            top,
            StateSet(-1f0, 1f0),
            Ising(c = ConstVal(0f0), b = 0f0, localpotential = 0f0);
            periodic = true,
        )
        return (; graph = g, algorithm = nothing, title = "Scenario: Hexagonal Graph")
    elseif name === :example_3d_langevin_coulomb
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

"""
    open_scenario_debug!(name = :example_3d_langevin_coulomb; start_process = true, trace = true, kwargs...)

Open one example-like `interface(g)` scenario and store the returned host in
`CLOSE_DEBUG_LAST`. Close tracing is enabled by default because these scenarios
are meant to catch intermittent macOS close stalls on realistic windows.
"""
function open_scenario_debug!(name::Symbol = :example_3d_langevin_coulomb; start_process = true, trace = true, kwargs...)
    name in CLOSE_DEBUG_SCENARIOS || throw(ArgumentError("Unknown close-debug scenario $name. Use CLOSE_DEBUG_SCENARIOS."))
    scenario = make_close_debug_scenario(name)
    println("Opening close-debug scenario: ", name)
    host = interface(scenario.graph; title = scenario.title, kwargs...)
    CLOSE_DEBUG_LAST[] = host
    trace === false || trace_last_close!(trace)
    if start_process
        if isnothing(scenario.algorithm)
            createProcess(scenario.graph)
        else
            createProcess(scenario.graph, scenario.algorithm)
        end
    end
    return nothing
end

"""
    repeat_scenario_debug!(name = :example_3d_langevin_coulomb; repeats = 10, start_process = true)

Prepare a repeated run of one realistic `interface(g)` scenario. Call
`open_next_scenario_repeat!()`, close the window, then repeat until complete.
"""
function repeat_scenario_debug!(name::Symbol = :example_3d_langevin_coulomb; repeats = 10, start_process = true)
    name in CLOSE_DEBUG_SCENARIOS || throw(ArgumentError("Unknown close-debug scenario $name. Use CLOSE_DEBUG_SCENARIOS."))
    CLOSE_DEBUG_SCENARIO_REPEAT_NAME[] = name
    CLOSE_DEBUG_SCENARIO_REPEAT_START_PROCESS[] = Bool(start_process)
    CLOSE_DEBUG_SCENARIO_REPEAT_LEFT[] = Int(repeats)
    CLOSE_DEBUG_SCENARIO_REPEAT_TOTAL[] = Int(repeats)
    println("Prepared scenario close-debug run for ", name, " x ", repeats)
    return nothing
end

"""
    open_next_scenario_repeat!(; trace = true, kwargs...)

Open the next scenario window prepared by `repeat_scenario_debug!`.
"""
function open_next_scenario_repeat!(; trace = true, kwargs...)
    if CLOSE_DEBUG_SCENARIO_REPEAT_LEFT[] <= 0
        println("Scenario close-debug run complete.")
        return nothing
    end
    iteration = CLOSE_DEBUG_SCENARIO_REPEAT_TOTAL[] - CLOSE_DEBUG_SCENARIO_REPEAT_LEFT[] + 1
    CLOSE_DEBUG_SCENARIO_REPEAT_LEFT[] -= 1
    println(
        "Scenario close-debug ",
        iteration,
        "/",
        CLOSE_DEBUG_SCENARIO_REPEAT_TOTAL[],
        ": ",
        CLOSE_DEBUG_SCENARIO_REPEAT_NAME[],
    )
    return open_scenario_debug!(
        CLOSE_DEBUG_SCENARIO_REPEAT_NAME[];
        start_process = CLOSE_DEBUG_SCENARIO_REPEAT_START_PROCESS[],
        trace,
        kwargs...,
    )
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
println("To repeat one realistic scenario, call repeat_scenario_debug!(); open_next_scenario_repeat!().")
