ENV["ISING_XOR_2IN_STATE"] = "continuous"
ENV["ISING_XOR_2IN_DYNAMICS"] = "langevin"
ENV["ISING_XOR_2IN_INIT_MODE"] = "zero"
ENV["ISING_XOR_2IN_TEMP"] = "0.001"
ENV["ISING_XOR_2IN_STEPSIZE"] = "0.1"
ENV["ISING_XOR_2IN_BETA"] = "1.0"
ENV["ISING_XOR_2IN_LR"] = "0.01"
ENV["ISING_XOR_2IN_WEIGHT_DECAY"] = "0"
ENV["ISING_XOR_2IN_WEIGHT_SCALE"] = "0.2"
ENV["ISING_XOR_2IN_BIAS_SCALE"] = "0.05"
ENV["ISING_XOR_2IN_WEIGHT_SEED"] = "13"
ENV["ISING_XOR_2IN_BIAS_SEED"] = "23"
ENV["ISING_XOR_2IN_BASE_SEED"] = "87300"
ENV["ISING_XOR_2IN_FREE_RELAXATION"] = "1000"
ENV["ISING_XOR_2IN_NUDGED_RELAXATION"] = "1000"
ENV["ISING_XOR_2IN_MINIT"] = "4"
ENV["ISING_XOR_2IN_EVAL_REPEATS"] = "32"
get!(ENV, "ISING_XOR_LIVE_WARMSTART", "true")
get!(
    ENV,
    "ISING_XOR_LIVE_WARMSTART_PATH",
    joinpath(
        @__DIR__,
        "..",
        "ext",
        "IsingLearning",
        "runs",
        "xor_2input_langevin_zero_T0001_s010_lr001_nodecay",
        "xor_statistical_ep_2input_trained_graph.jld2",
    ),
)
ENV["ISING_XOR_LIVE_START_LEARNING"] = "true"

const XOR_TRAINING_SOURCE = joinpath(@__DIR__, "..", "ext", "IsingLearning", "examples", "xor_statistical_ep_2input.jl")
include_string(
    @__MODULE__,
    replace(read(XOR_TRAINING_SOURCE, String), r"\nmain\(\)\s*$" => "\n"),
    XOR_TRAINING_SOURCE,
)

using GLMakie
using IsingLearning.InteractiveIsing.Windows
using Printf

GLMakie.activate!()

const LIVE_EVAL_EVERY = parse(Int, get(ENV, "ISING_XOR_LIVE_EVAL_EVERY", "1"))
const LIVE_EPOCHS_PER_TICK = parse(Int, get(ENV, "ISING_XOR_LIVE_EPOCHS_PER_TICK", "1"))
const LIVE_DISPLAY_STEPSIZE = parse(FT, get(ENV, "ISING_XOR_LIVE_DISPLAY_STEPSIZE", string(STEPSIZE)))
const LIVE_WARMSTART = parse(Bool, get(ENV, "ISING_XOR_LIVE_WARMSTART", "true"))
const LIVE_WARMSTART_PATH = normpath(get(ENV, "ISING_XOR_LIVE_WARMSTART_PATH", ""))
const LIVE_START_LEARNING = parse(Bool, get(ENV, "ISING_XOR_LIVE_START_LEARNING", "true"))

function load_live_warmstart!(graph)
    LIVE_WARMSTART || return false
    isfile(LIVE_WARMSTART_PATH) || return false
    saved_graph = II.load_isinggraph(LIVE_WARMSTART_PATH)
    IsingLearning.sync_graph_params!(graph, IsingLearning.read_graph_params(saved_graph))
    return true
end

function xor_learning_graph()
    input_layer = II.Layer(
        1, 2,
        II.StateSet(-one(FT), one(FT)),
        II.Continuous(),
        II.Coords(0, 0, 0);
        periodic = false,
    )
    hidden_layer = II.Layer(
        4, 4,
        II.StateSet(-one(FT), one(FT)),
        II.Continuous(),
        II.Coords(0, 3, 0);
        periodic = false,
    )
    output_layer = II.Layer(
        1, 2,
        II.StateSet(-one(FT), one(FT)),
        II.Continuous(),
        II.Coords(0, 8, 0);
        periodic = false,
    )

    wg = signed_weight_generator()
    clamping_target = g -> II.filltype(Vector, zero(FT), II.statelen(g))
    clamping_beta = II.UniformArray(zero(FT))
    hamiltonian = II.Bilinear() + II.MagField(b = bias_generator()) +
        II.Clamping(β = clamping_beta, y = clamping_target)

    graph = II.IsingGraph(
        input_layer,
        wg,
        hidden_layer,
        deepcopy(wg),
        output_layer,
        hamiltonian;
        precision = FT,
        index_set = g -> II.ToggledIndexSet(g),
    )
    II.temp!(graph, TEMP)
    II.off!(graph.index_set, 1)
    return graph
end

function xor_learning_layer(graph)
    base_dynamics = II.BlockLangevin(
        stepsize = STEPSIZE,
        adjusted = false,
        block_size = BLOCK_SIZE,
        group_steps = 1,
    )
    free_dynamics = deepcopy(base_dynamics)
    return LayeredIsingGraphLayer(
        () -> xor_learning_graph();
        input_idxs = II.layerrange(graph[1]),
        output_idxs = II.layerrange(graph[end]),
        β = BETA,
        fullsweeps = 1,
        relaxation_steps = FREE_RELAXATION,
        free_relaxation_steps = FREE_RELAXATION,
        nudged_relaxation_steps = NUDGED_RELAXATION,
        dynamics_algorithm = free_dynamics,
        nudged_dynamics_algorithm = deepcopy(base_dynamics),
        validation_algorithm = deepcopy(base_dynamics),
    )
end

function apply_xor_case_zero!(graph, a::Bool, b::Bool)
    fill!(II.state(graph), zero(FT))
    II.state(graph[1]) .= reshape(xor_input(a, b), size(II.state(graph[1])))
    II.off!(graph.index_set, 1)
    return graph
end

function live_output_status(graph, a::Bool, b::Bool)
    out = collect(vec(II.state(graph[end])))
    target = xor_target(a, b)
    prediction = out[2] > out[1]
    mse = sum(abs2, out .- target) / FT(length(target))
    return @sprintf(
        "input = (%d, %d)    xor = %d    output = [% .3f, % .3f]    prediction = %d    MSE = %.4f",
        Int(a),
        Int(b),
        Int(xor(a, b)),
        out[1],
        out[2],
        Int(prediction),
        mse,
    )
end

function restart_live_process!(graph, dynamics, a::Bool, b::Bool)
    Processes.close(graph)
    apply_xor_case_zero!(graph, a, b)
    return II.createProcess(graph, dynamics)
end

function make_live_trainer(graph)
    layer = xor_learning_layer(graph)
    trainer = init_xor_trainer(layer; graph, optimiser = Optimisers.Adam(LEARNING_RATE))
    set_trainer_temperature!(trainer, TEMP)
    return trainer
end

function live_metric_row(epoch, metrics, grad_metrics, trainer, initial_params)
    param_delta = sqrt(
        sum(abs2, trainer.params.w .- initial_params.w) +
        sum(abs2, trainer.params.b .- initial_params.b)
    )
    return (;
        epoch = FT(epoch),
        mse = FT(metrics.mse),
        accuracy = FT(metrics.accuracy),
        grad_norm = FT(grad_metrics.grad_norm),
        response_norm = FT(grad_metrics.response_norm),
        param_delta = FT(param_delta),
    )
end

function push_live_metric!(state, row)
    lock(state.lock) do
        push!(state.epochs, row.epoch)
        push!(state.mses, row.mse)
        push!(state.accuracies, row.accuracy)
        if row.mse < state.best_mse[]
            state.best_mse[] = row.mse
            state.best_epoch[] = Int(row.epoch)
            state.best_params[] = deepcopy(state.trainer.params)
        end
        state.latest[] = row
    end
    return row
end

function mse_line_getter(state)
    lock(state.lock) do
        return copy(state.epochs), copy(state.mses)
    end
end

function learning_status(state)
    row = state.latest[]
    source = state.warmstarted[] ? "warm-start" : "random-start"
    return @sprintf(
        "%s    learning %s    epoch = %d    eval MSE = %.5f    acc = %.2f    best = %.5f @ %d    grad = %.3f    response = %.3f",
        source,
        state.learning[] ? "on" : "off",
        Int(row.epoch),
        row.mse,
        row.accuracy,
        state.best_mse[],
        state.best_epoch[],
        row.grad_norm,
        row.response_norm,
    )
end

function start_learning_loop!(state)
    return @async begin
        while state.open[]
            if !state.learning[]
                sleep(0.05)
                continue
            end

            grad_metrics = (; grad_norm = zero(FT), response_norm = zero(FT))
            for _ in 1:LIVE_EPOCHS_PER_TICK
                state.epoch[] += 1
                grad_metrics = train_epoch!(
                    state.trainer,
                    state.x,
                    state.y,
                    state.batch_gradient,
                    state.epoch[],
                )
            end

            if state.epoch[] == 1 || state.epoch[] % LIVE_EVAL_EVERY == 0
                metrics = evaluate_xor!(
                    state.trainer,
                    state.x,
                    state.y;
                    seed_offset = BASE_SEED + EVAL_SEED_OFFSET,
                )
                push_live_metric!(
                    state,
                    live_metric_row(state.epoch[], metrics, grad_metrics, state.trainer, state.initial_params),
                )
            end

            yield()
        end
    end
end

function xor_interactive_learning()
    graph = xor_learning_graph()
    warmstarted = load_live_warmstart!(graph)
    x, y = xor_dataset()
    trainer = make_live_trainer(graph)
    batch_gradient = IsingLearning.gradient_buffer(graph)
    initial_params = deepcopy(trainer.params)

    zero_grad = (; grad_norm = zero(FT), response_norm = zero(FT))
    initial_metrics = evaluate_xor!(trainer, x, y; seed_offset = BASE_SEED + EVAL_SEED_OFFSET)
    initial_row = live_metric_row(0, initial_metrics, zero_grad, trainer, initial_params)

    state = (;
        graph,
        trainer,
        x,
        y,
        batch_gradient,
        initial_params,
        epoch = Ref(0),
        learning = Observable(LIVE_START_LEARNING),
        latest = Observable(initial_row),
        best_mse = Ref(initial_row.mse),
        best_epoch = Ref(0),
        best_params = Ref(deepcopy(trainer.params)),
        warmstarted = Ref(warmstarted),
        open = Observable(true),
        lock = ReentrantLock(),
        epochs = FT[],
        mses = FT[],
        accuracies = FT[],
    )
    push_live_metric!(state, initial_row)

    display_dynamics = II.BlockLangevin(
        stepsize = LIVE_DISPLAY_STEPSIZE,
        adjusted = false,
        block_size = BLOCK_SIZE,
        group_steps = 1,
    )

    bit_a = Observable(false)
    bit_b = Observable(false)
    status = Observable(live_output_status(graph, bit_a[], bit_b[]))
    learn_status = lift((_, __) -> learning_status(state), state.latest, state.learning)

    host = window(title = "Interactive XOR Learning", size = (1200, 950), fps = 30, polling_rate = 10)

    controls = GridLayout(host.figure[1, 1])
    button_a = Button(
        controls[1, 1],
        label = lift(a -> "x1 = $(Int(a))", bit_a),
        width = 110,
        height = 34,
    )
    button_b = Button(
        controls[1, 2],
        label = lift(b -> "x2 = $(Int(b))", bit_b),
        width = 110,
        height = 34,
    )
    learning_button = Button(
        controls[1, 3],
        label = lift(on -> on ? "Pause Learning" : "Resume Learning", state.learning),
        width = 150,
        height = 34,
    )
    restore_button = Button(
        controls[1, 4],
        label = "Restore Best",
        width = 120,
        height = 34,
    )
    Label(controls[1, 5], status; tellwidth = false, halign = :left)
    Label(controls[2, 1:5], learn_status; tellwidth = false, halign = :left)

    layer_panel = panel!(
        host,
        AllLayersViewPanel(
            graph;
            colormap = :balance,
            labels = true,
            axis_kwargs = (title = "live source graph: input | hidden | output",),
        ),
        (2, 1),
    )
    mse_panel = panel!(
        host,
        InteractiveLinesPanel(
            () -> mse_line_getter(state);
            xlabel = "epoch",
            ylabel = "MSE",
            title = "live validation MSE",
            line_kwargs = (; color = :dodgerblue, linewidth = 2),
            update_rate = 5,
        ),
        (3, 1),
    )

    process_ref = Ref{Any}(restart_live_process!(graph, display_dynamics, bit_a[], bit_b[]))
    learning_task = start_learning_loop!(state)

    function set_case!(a::Bool, b::Bool)
        bit_a[] = a
        bit_b[] = b
        process_ref[] = restart_live_process!(graph, display_dynamics, a, b)
        status[] = live_output_status(graph, a, b)
        return nothing
    end

    register!(host, on(button_a.clicks) do _
        set_case!(!bit_a[], bit_b[])
    end)
    register!(host, on(button_b.clicks) do _
        set_case!(bit_a[], !bit_b[])
    end)
    register!(host, on(learning_button.clicks) do _
        state.learning[] = !state.learning[]
    end)
    register!(host, on(restore_button.clicks) do _
        state.learning[] = false
        lock(state.lock) do
            trainer.params = deepcopy(state.best_params[])
            IsingLearning._broadcast_params!(trainer)
        end
        process_ref[] = restart_live_process!(graph, display_dynamics, bit_a[], bit_b[])
        status[] = live_output_status(graph, bit_a[], bit_b[])
        return nothing
    end)
    register_frame!(host) do _
        status[] = live_output_status(graph, bit_a[], bit_b[])
        return nothing
    end
    register!(host, on(host.open) do isopen
        isopen && return
        state.open[] = false
        Processes.close(graph)
        IsingLearning.close_trainer!(trainer)
    end)

    return (; graph, host, layer_panel, mse_panel, process = process_ref, trainer, learning_task, state)
end

if get(ENV, "INTERACTIVE_XOR_LEARNING_HEADLESS", "false") == "true"
    graph = xor_learning_graph()
    load_live_warmstart!(graph)
    xor_learning_demo = (; graph)
else
    xor_learning_demo = xor_interactive_learning()
end
