const CHECKERBOARD_XOR_SOURCE = joinpath(
    @__DIR__,
    "..",
    "ext",
    "IsingLearning",
    "experiments",
    "local_checkerboard_xor",
    "local_checkerboard_xor.jl",
)

include(CHECKERBOARD_XOR_SOURCE)

using GLMakie
using IsingLearning.InteractiveIsing.Windows
using Printf
using Random

GLMakie.activate!()

const CHECKERBOARD_XOR_DEFAULT_GRAPH = normpath(joinpath(
    @__DIR__,
    "..",
    "ext",
    "IsingLearning",
    "experiments",
    "local_checkerboard_xor",
    "runs",
    "metropolis_2x2_sym_lazy_T0010_inter010_20260510_000037",
    "checker_2x2_global",
    "checker_2x2_global_best_graph.jld2",
))

const CHECKERBOARD_XOR_GRAPH_PATH = normpath(get(ENV, "CHECKERBOARD_XOR_GRAPH_PATH", CHECKERBOARD_XOR_DEFAULT_GRAPH))
checkerboard_demo_init() = Symbol(get(ENV, "CHECKERBOARD_XOR_INIT", "random"))

"""
    checkerboard_demo_config()

Return the 2x2 local checkerboard XOR configuration used when no saved graph is
available. The defaults match the best true-symmetric Metropolis run so far.
"""
function checkerboard_demo_config()
    return LocalCheckerboardConfig(
        name = "checker_2x2_global",
        side = 2,
        hidden_side = 2,
        code_side = 2,
        code_stride = 1,
        epochs = 0,
        log_every = 1,
        minit = 8,
        eval_repeats = 64,
        workers = 1,
        free_relaxation = 150,
        nudged_relaxation = 150,
        β = FT(0.2),
        lr = FT(0.003),
        weight_decay = zero(FT),
        temp = FT(0.01),
        temp_is_factor = false,
        inter_weight_scale = FT(0.10),
        input_internal_scale = FT(0.0141421356237),
        hidden_internal_scale = FT(0.0141421356237),
        output_internal_scale = FT(0.0141421356237),
        dynamics_mode = :metropolis,
        state_mode = :discrete,
        init_mode = checkerboard_demo_init(),
    )
end

"""
    checkerboard_demo_graph()

Load the trained checkerboard XOR graph from `CHECKERBOARD_XOR_GRAPH_PATH`, or
construct a fresh graph with `checkerboard_demo_config()` when the file is not
present.
"""
function checkerboard_demo_graph()
    if isfile(CHECKERBOARD_XOR_GRAPH_PATH)
        graph = II.load_isinggraph(CHECKERBOARD_XOR_GRAPH_PATH)
        II.temp!(graph, effective_temp(graph, get(graph.addons, :checker_config, checkerboard_demo_config())))
        checkerboard_demo_layout!(graph)
        checkerboard_reset_state!(graph)
        return graph
    end

    @warn "CHECKERBOARD_XOR_GRAPH_PATH not found; using an untrained graph" CHECKERBOARD_XOR_GRAPH_PATH
    graph = checkerboard_graph(checkerboard_demo_config())
    checkerboard_demo_layout!(graph)
    checkerboard_reset_state!(graph)
    return graph
end

"""
    checkerboard_demo_layout!(graph)

Place the input, hidden, and output layers side by side for `AllLayersViewPanel`.
The saved training graph uses coordinates for connection geometry, not for the
global image layout, so the raw layer rectangles can overlap in the display.
"""
function checkerboard_demo_layout!(graph)
    x = 0
    for layer in II.layers(graph)
        II.setcoords!(layer; y = 0, x = x, z = 0)
        x += size(layer, 2) + 2
    end
    return graph
end

function checkerboard_demo_dynamics(graph)
    config = get(graph.addons, :checker_config, checkerboard_demo_config())
    if config.dynamics_mode === :metropolis
        return II.IsingMetropolis()
    elseif config.dynamics_mode === :langevin
        return II.BlockLangevin(
            stepsize = config.stepsize,
            adjusted = false,
            block_size = config.block_size,
            group_steps = 1,
        )
    else
        error("Unsupported checkerboard demo dynamics mode $(config.dynamics_mode)")
    end
end

function checkerboard_has_discrete_layers(graph)
    return any(layer -> II.statetype(layer) isa II.Discrete, II.layers(graph))
end

function checkerboard_repair_discrete_state!(graph)
    rng = Random.default_rng()
    s = II.state(graph)
    for layer in II.layers(graph)
        II.statetype(layer) isa II.Discrete || continue
        states = II.stateset(layer)
        for idx in II.graphidxs(layer)
            s[idx] in states && continue
            s[idx] = rand(rng, states)
        end
    end
    return graph
end

function checkerboard_reset_state!(graph, init_mode::Symbol = checkerboard_demo_init())
    s = II.state(graph)
    if init_mode === :zero
        if checkerboard_has_discrete_layers(graph)
            s .= II.initRandomState(graph)
        else
            fill!(s, zero(eltype(s)))
        end
    elseif init_mode === :random
        s .= II.initRandomState(graph)
    else
        throw(ArgumentError("CHECKERBOARD_XOR_INIT must be zero or random, got $init_mode"))
    end
    checkerboard_repair_discrete_state!(graph)
    return graph
end

function checkerboard_apply_case!(graph, a::Bool, b::Bool)
    config = get(graph.addons, :checker_config, checkerboard_demo_config())
    checkerboard_reset_state!(graph)
    apply_checker_input!(graph, FT[a ? 1 : -1, b ? 1 : -1], config)
    checkerboard_repair_discrete_state!(graph)
    return graph
end

function checkerboard_score(graph)
    output_idxs = graph.addons[:checker_output_idxs]
    readout = graph.addons[:checker_readout]
    return sum(II.state(graph)[idx] * w for (idx, w) in zip(output_idxs, readout))
end

checkerboard_target(a::Bool, b::Bool) = xor_target(a, b)

function checkerboard_case_status(graph, a::Bool, b::Bool)
    config = get(graph.addons, :checker_config, checkerboard_demo_config())
    case_idx = case_index(a, b)
    frozen = length(graph.addons[:checker_frozen_sets][case_idx])
    score = checkerboard_score(graph)
    target = checkerboard_target(a, b)
    prediction = score > zero(score)
    mse = abs2(score - target)
    invalid = sum(
        count(v -> !(v in II.stateset(layer)), vec(II.state(layer)))
        for layer in II.layers(graph)
        if II.statetype(layer) isa II.Discrete
    )
    return @sprintf(
        "A=%d  B=%d    XOR=%d    frozen=%d    score=% .3f    target=% .1f    prediction=%d    MSE=%.4f    invalid=%d    T=%.4g",
        Int(a),
        Int(b),
        Int(xor(a, b)),
        frozen,
        score,
        target,
        Int(prediction),
        mse,
        invalid,
        II.temp(graph),
    )
end

function checkerboard_restart_process!(graph, dynamics, a::Bool, b::Bool)
    Processes.close(graph)
    checkerboard_apply_case!(graph, a, b)
    checkerboard_repair_discrete_state!(graph)
    return II.createProcess(graph, dynamics)
end

"""
    checkerboard_xor_interactive()

Open a running, non-learning checkerboard XOR demo. The buttons toggle the two
physical input masks. A selected bit freezes one checkerboard mask in the input
layer to `+1`; `(0, 0)` freezes no input spins.
"""
function checkerboard_xor_interactive()
    graph = checkerboard_demo_graph()
    dynamics = checkerboard_demo_dynamics(graph)

    bit_a = Observable(false)
    bit_b = Observable(false)
    status = Observable(checkerboard_case_status(graph, bit_a[], bit_b[]))

    host = window(
        title = "Interactive Checkerboard XOR",
        size = (1250, 850),
        fps = 30,
        polling_rate = 10,
    )
    controls = GridLayout(host.figure[1, 1])

    button_a = Button(
        controls[1, 1],
        label = lift(a -> "A = $(Int(a))", bit_a),
        width = 110,
        height = 34,
    )
    button_b = Button(
        controls[1, 2],
        label = lift(b -> "B = $(Int(b))", bit_b),
        width = 110,
        height = 34,
    )
    reset_button = Button(controls[1, 3], label = "Reset", width = 110, height = 34)
    Label(controls[1, 4], status; tellwidth = false, halign = :left)

    panel = panel!(
        host,
        AllLayersViewPanel(
            graph;
            colormap = :balance,
            labels = true,
            axis_kwargs = (title = "local checkerboard XOR: input | hidden | output",),
        ),
        (2, 1),
    )

    process_ref = Ref{Any}(checkerboard_restart_process!(graph, dynamics, bit_a[], bit_b[]))

    function set_case!(a::Bool, b::Bool)
        bit_a[] = a
        bit_b[] = b
        process_ref[] = checkerboard_restart_process!(graph, dynamics, a, b)
        status[] = checkerboard_case_status(graph, a, b)
        return nothing
    end

    register!(host, on(button_a.clicks) do _
        set_case!(!bit_a[], bit_b[])
    end)
    register!(host, on(button_b.clicks) do _
        set_case!(bit_a[], !bit_b[])
    end)
    register!(host, on(reset_button.clicks) do _
        set_case!(bit_a[], bit_b[])
    end)
    register_frame!(host) do _
        status[] = checkerboard_case_status(graph, bit_a[], bit_b[])
        return nothing
    end
    onclose!(host) do _
        Processes.close(graph)
    end

    return (; graph, host, panel, process = process_ref, bit_a, bit_b)
end

if get(ENV, "CHECKERBOARD_XOR_HEADLESS", "false") == "true"
    checkerboard_xor_demo = (; graph = checkerboard_demo_graph())
else
    checkerboard_xor_demo = checkerboard_xor_interactive()
end
