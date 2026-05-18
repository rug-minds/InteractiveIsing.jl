using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using IsingLearning
using Optimisers
using Random
using Statistics
using Dates
using CairoMakie
import IsingLearning.InteractiveIsing: @WG, WeightGenerator

const II = IsingLearning.InteractiveIsing
const Processes = II.Processes
const FT = Float64

const EDGE_CASES = ((false, false), (false, true), (true, false), (true, true))
const EDGE_HEIGHT_REF = Ref(8)
const EDGE_WIDTH_REF = Ref(8)
const EDGE_INPUT_SCALE_REF = Ref{FT}(0.25)
const EDGE_HIDDEN_SCALE_REF = Ref{FT}(0.04)
const EDGE_OUTPUT_SCALE_REF = Ref{FT}(0.25)
const EDGE_OUTPUT_INTERNAL_SCALE_REF = Ref{FT}(0.0)
const EDGE_HIDDEN_NN_REF = Ref(1)
const EDGE_INPUT_RNG_REF = Ref(Random.MersenneTwister(1))
const EDGE_HIDDEN_RNG_REF = Ref(Random.MersenneTwister(2))
const EDGE_OUTPUT_RNG_REF = Ref(Random.MersenneTwister(3))

"""Configuration for the edge-connected `2 -> 8x8 -> 2` XOR experiment."""
Base.@kwdef struct EdgeTwoOutConfig
    epochs::Int = parse(Int, get(ENV, "EDGE_TWOOUT_EPOCHS", "3000"))
    log_every::Int = parse(Int, get(ENV, "EDGE_TWOOUT_LOG_EVERY", "250"))
    minit::Int = parse(Int, get(ENV, "EDGE_TWOOUT_MINIT", "4"))
    eval_repeats::Int = parse(Int, get(ENV, "EDGE_TWOOUT_EVAL_REPEATS", "32"))
    workers::Int = parse(Int, get(ENV, "EDGE_TWOOUT_WORKERS", "1"))
    free_relaxation::Int = parse(Int, get(ENV, "EDGE_TWOOUT_FREE", "1000"))
    nudged_relaxation::Int = parse(Int, get(ENV, "EDGE_TWOOUT_NUDGED", "1000"))
    β::FT = parse(FT, get(ENV, "EDGE_TWOOUT_BETA", "1.0"))
    lr::FT = parse(FT, get(ENV, "EDGE_TWOOUT_LR", "0.005"))
    weight_decay::FT = parse(FT, get(ENV, "EDGE_TWOOUT_WEIGHT_DECAY", "0.0"))
    temp::FT = parse(FT, get(ENV, "EDGE_TWOOUT_TEMP", "0.001"))
    validation_temp::FT = parse(FT, get(ENV, "EDGE_TWOOUT_VALIDATION_TEMP", get(ENV, "EDGE_TWOOUT_TEMP", "0.001")))
    stepsize::FT = parse(FT, get(ENV, "EDGE_TWOOUT_STEPSIZE", "0.10"))
    dynamics::Symbol = Symbol(get(ENV, "EDGE_TWOOUT_DYNAMICS", "block"))
    block_size::Int = parse(Int, get(ENV, "EDGE_TWOOUT_BLOCK_SIZE", "8"))
    hidden_height::Int = parse(Int, get(ENV, "EDGE_TWOOUT_HEIGHT", "4"))
    hidden_width::Int = parse(Int, get(ENV, "EDGE_TWOOUT_WIDTH", "4"))
    hidden_nn::Int = parse(Int, get(ENV, "EDGE_TWOOUT_NN", "5"))
    output_repeats::Int = parse(Int, get(ENV, "EDGE_TWOOUT_OUTPUT_REPEATS", "1"))
    input_scale::FT = parse(FT, get(ENV, "EDGE_TWOOUT_INPUT_SCALE", "0.40"))
    hidden_scale::FT = parse(FT, get(ENV, "EDGE_TWOOUT_HIDDEN_SCALE", "0.20"))
    output_scale::FT = parse(FT, get(ENV, "EDGE_TWOOUT_OUTPUT_SCALE", "0.40"))
    output_internal_scale::FT = parse(FT, get(ENV, "EDGE_TWOOUT_OUTPUT_INTERNAL_SCALE", "0.0"))
    gradient_mode::Symbol = Symbol(get(ENV, "EDGE_TWOOUT_GRADIENT_MODE", "symmetric"))
    bias_scale::FT = parse(FT, get(ENV, "EDGE_TWOOUT_BIAS_SCALE", "0.05"))
    weight_seed::Int = parse(Int, get(ENV, "EDGE_TWOOUT_WEIGHT_SEED", "1701"))
    bias_seed::Int = parse(Int, get(ENV, "EDGE_TWOOUT_BIAS_SEED", "1703"))
    base_seed::Int = parse(Int, get(ENV, "EDGE_TWOOUT_BASE_SEED", "917000"))
end

"""Accumulate `dH(s_a)/dθ - dH(s_b)/dθ` into an existing gradient buffer."""
function edge_difference_gradient!(graph, s_a, s_b, buffers)
    bilinear = graph.hamiltonian[II.Bilinear]
    magfield = graph.hamiltonian[II.MagField]
    II.parameter_derivative(bilinear, s_a, dJ = buffers.w, buffermode = II.AccumulateBuffer{+}())
    II.parameter_derivative(bilinear, s_b, dJ = buffers.w, buffermode = II.SubtractBuffer())
    II.parameter_derivative(magfield, s_a, db = buffers.b, buffermode = II.AccumulateBuffer{+}())
    II.parameter_derivative(magfield, s_b, db = buffers.b, buffermode = II.SubtractBuffer())
    return buffers
end

"""Return bipolar two-input XOR samples and two-spin bipolar output targets."""
function edge_twoout_dataset(config::EdgeTwoOutConfig = EdgeTwoOutConfig())
    x = Matrix{FT}(undef, 2, 4)
    y = Matrix{FT}(undef, 2 * config.output_repeats, 4)
    for (col, (a, b)) in enumerate(EDGE_CASES)
        x[:, col] .= (a ? one(FT) : -one(FT), b ? one(FT) : -one(FT))
        class_values = xor(a, b) ? (-one(FT), one(FT)) : (one(FT), -one(FT))
        for repeat_idx in 1:config.output_repeats
            y[repeat_idx, col] = class_values[1]
            y[config.output_repeats + repeat_idx, col] = class_values[2]
        end
    end
    return x, y
end

"""True if a hidden coordinate is on the left edge."""
edge_left(c) = c[2] == 1

"""True if a hidden coordinate is on the right edge."""
edge_right(c) = c[2] == EDGE_WIDTH_REF[]

"""Input spin 1 drives the upper half of the left edge; spin 2 drives the lower half."""
function edge_input_weight(; c1, c2)
    edge_left(c2) || return zero(FT)
    upper = c2[1] <= EDGE_HEIGHT_REF[] ÷ 2
    row = c1[1]
    if (row == 1 && upper) || (row == 2 && !upper)
        return EDGE_INPUT_SCALE_REF[] * abs(randn(EDGE_INPUT_RNG_REF[], FT))
    end
    return zero(FT)
end

"""Random learned local hidden coupling within `EDGE_HIDDEN_NN_REF` grid distance."""
function edge_hidden_weight(; dr)
    dr <= EDGE_HIDDEN_NN_REF[] || return zero(FT)
    return EDGE_HIDDEN_SCALE_REF[] * randn(EDGE_HIDDEN_RNG_REF[], FT)
end

"""Every right-edge hidden spin connects to every output-code spin with signed weights."""
function edge_output_weight(; c1, c2)
    edge_right(c1) || return zero(FT)
    return EDGE_OUTPUT_SCALE_REF[] * randn(EDGE_OUTPUT_RNG_REF[], FT)
end

"""Couple repeated output-code spins so each class group votes coherently."""
function edge_output_internal_weight(; c1, c2)
    scale = EDGE_OUTPUT_INTERNAL_SCALE_REF[]
    iszero(scale) && return zero(FT)
    repeats = max(1, EDGE_OUTPUT_REPEATS_REF[])
    class1 = c1[1] <= repeats
    class2 = c2[1] <= repeats
    return class1 == class2 ? scale : -scale
end

"""Create the input-to-hidden edge generator."""
function edge_input_generator(config::EdgeTwoOutConfig)
    EDGE_INPUT_SCALE_REF[] = config.input_scale
    EDGE_INPUT_RNG_REF[] = Random.MersenneTwister(config.weight_seed)
    return @WG edge_input_weight NN = :all symmetric = true
end

"""Create the hidden-layer local generator."""
function edge_hidden_generator(config::EdgeTwoOutConfig)
    EDGE_HIDDEN_SCALE_REF[] = config.hidden_scale
    EDGE_HIDDEN_NN_REF[] = config.hidden_nn
    EDGE_HIDDEN_RNG_REF[] = Random.MersenneTwister(config.weight_seed + 1)
    config.hidden_nn == 1 && return @WG edge_hidden_weight NN = 1 symmetric = true
    config.hidden_nn == 2 && return @WG edge_hidden_weight NN = 2 symmetric = true
    config.hidden_nn == 3 && return @WG edge_hidden_weight NN = 3 symmetric = true
    config.hidden_nn == 5 && return @WG edge_hidden_weight NN = 5 symmetric = true
    config.hidden_nn == 8 && return @WG edge_hidden_weight NN = 8 symmetric = true
    throw(ArgumentError("supported EDGE_TWOOUT_NN values are 1, 2, 3, 5, and 8"))
end

"""Create the hidden-to-output edge generator."""
function edge_output_generator(config::EdgeTwoOutConfig)
    EDGE_OUTPUT_SCALE_REF[] = config.output_scale
    EDGE_OUTPUT_RNG_REF[] = Random.MersenneTwister(config.weight_seed + 2)
    return @WG edge_output_weight NN = :all symmetric = true
end

const EDGE_OUTPUT_REPEATS_REF = Ref(1)

"""Create optional structured couplings inside the repeated output code."""
function edge_output_internal_generator(config::EdgeTwoOutConfig)
    EDGE_OUTPUT_REPEATS_REF[] = config.output_repeats
    EDGE_OUTPUT_INTERNAL_SCALE_REF[] = config.output_internal_scale
    return @WG edge_output_internal_weight NN = 4 symmetric = true
end

"""Build the edge-connected graph with trainable bilinear weights and biases only."""
function edge_twoout_graph(config::EdgeTwoOutConfig)
    EDGE_HEIGHT_REF[] = config.hidden_height
    EDGE_WIDTH_REF[] = config.hidden_width
    rng_b = Random.MersenneTwister(config.bias_seed)
    input = II.Layer(2, 1, II.StateSet(-one(FT), one(FT)), II.Continuous(), II.Coords(0, 0, 0))
    hidden = II.Layer(
        config.hidden_height,
        config.hidden_width,
        II.StateSet(-one(FT), one(FT)),
        edge_hidden_generator(config),
        II.Continuous(),
        II.Coords(0, 0, 0);
        periodic = false,
    )
    output = II.Layer(
        2 * config.output_repeats,
        1,
        II.StateSet(-one(FT), one(FT)),
        edge_output_internal_generator(config),
        II.Continuous(),
        II.Coords(0, 0, 0);
        periodic = false,
    )
    b = g -> config.bias_scale .* randn(rng_b, FT, II.statelen(g))
    y = g -> II.filltype(Vector, zero(FT), II.statelen(g))
    mask = g -> II.filltype(Vector, zero(FT), II.statelen(g))
    hamiltonian = II.Bilinear() + II.MagField(b = b) +
        II.Clamping(β = II.UniformArray(zero(FT)), y = y, mask = mask)
    graph = II.IsingGraph(
        input,
        edge_input_generator(config),
        hidden,
        edge_output_generator(config),
        output,
        hamiltonian;
        precision = FT,
        index_set = g -> II.ToggledIndexSet(g),
    )
    II.temp!(graph, config.temp)
    return graph
end

"""Return the unadjusted block Langevin dynamics used for this edge experiment."""
function edge_twoout_dynamics(config::EdgeTwoOutConfig)
    config.dynamics === :block && return II.BlockLangevin(
        stepsize = config.stepsize,
        adjusted = false,
        block_size = config.block_size,
        group_steps = 1,
    )
    config.dynamics === :local && return II.LocalLangevin(
        stepsize = config.stepsize,
        max_drift_fraction = one(FT),
        adjusted = false,
        order = :random,
        group_steps = 1,
    )
    throw(ArgumentError("EDGE_TWOOUT_DYNAMICS must be block or local, got $(config.dynamics)"))
end

"""Wrap the graph in the existing IsingLearning layer abstraction."""
function edge_twoout_layer(graph, config::EdgeTwoOutConfig)
    dynamics = edge_twoout_dynamics(config)
    return LayeredIsingGraphLayer(
        () -> edge_twoout_graph(config);
        input_idxs = II.layerrange(graph[1]),
        output_idxs = II.layerrange(graph[end]),
        β = config.β,
        fullsweeps = 1,
        relaxation_steps = config.free_relaxation,
        free_relaxation_steps = config.free_relaxation,
        nudged_relaxation_steps = config.nudged_relaxation,
        dynamics_algorithm = deepcopy(dynamics),
        nudged_dynamics_algorithm = deepcopy(dynamics),
        validation_algorithm = deepcopy(dynamics),
    )
end

"""Set all spins to zero before applying the next input pattern."""
function edge_zero_init!(model)
    fill!(II.state(model), zero(eltype(model)))
    return model
end

"""Free-phase routine with zero initialization, matching the working simple XOR recipe."""
function edge_forward(layer, config::EdgeTwoOutConfig)
    dynamics_algorithm = edge_twoout_dynamics(config)
    n_units = layer.nunits
    steps = layer.free_relaxation_steps
    forward = Processes.@Routine begin
        @alias dynamics = dynamics_algorithm
        @state equilibrium_state = zeros(n_units)
        @state x

        edge_zero_init!(dynamics.model)
        IsingLearning.apply_input(dynamics.model, x)
        model = @repeat steps dynamics()
        IsingLearning.copyvector!(equilibrium_state, @transform(m -> II.state(m), model))
    end
    return (; algorithm = forward, dynamics = forward.dynamics)
end

"""Plus/minus nudged routines restored from the free endpoint."""
function edge_nudged(layer, config::EdgeTwoOutConfig)
    beta = layer.β
    steps = layer.nudged_relaxation_steps
    dynamics_algorithm = edge_twoout_dynamics(config)
    plus_capture = IsingLearning.Capturer()
    minus_capture = IsingLearning.Capturer()

    plus = Processes.@Routine begin
        @state equilibrium_state
        @state y
        @state x
        @state nudged_beta = beta
        @alias dynamics = dynamics_algorithm
        @alias plus_capture = plus_capture

        IsingLearning.setgraph!(isinggraph = dynamics.model, target = equilibrium_state)
        IsingLearning.apply_input(dynamics.model, x)
        IsingLearning.apply_targets(dynamics.model, y)
        IsingLearning.set_clamping_beta!(dynamics.model, nudged_beta)
        model = @repeat steps dynamics()
        plus_capture(isinggraph = model)
    end

    minus = Processes.@Routine begin
        @state equilibrium_state
        @state y
        @state x
        @state nudged_beta = -beta
        @alias dynamics = dynamics_algorithm
        @alias minus_capture = minus_capture

        IsingLearning.setgraph!(isinggraph = dynamics.model, target = equilibrium_state)
        IsingLearning.apply_input(dynamics.model, x)
        IsingLearning.apply_targets(dynamics.model, y)
        IsingLearning.set_clamping_beta!(dynamics.model, nudged_beta)
        model = @repeat steps dynamics()
        minus_capture(isinggraph = model)
    end

    final = Processes.@CompositeAlgorithm begin
        @input clamping_beta = beta
        @state buffers
        @alias plus = plus
        @alias minus = minus
        plus.nudged_beta = clamping_beta
        @context c1 = plus()
        minus.nudged_beta = @transform(x -> -x, clamping_beta)
        @context c2 = minus()
    end
    return (; algorithm = final, plus, minus, plus_capture, minus_capture, dynamics = plus.dynamics)
end

"""Full zero-init EqProp composite for this edge graph."""
function edge_forward_and_nudged(layer, config::EdgeTwoOutConfig)
    forward = edge_forward(layer, config).algorithm
    nudged = edge_nudged(layer, config)
    beta = layer.β
    final = if config.gradient_mode == :symmetric
        Processes.@CompositeAlgorithm begin
            @input clamping_beta = beta
            @state buffers
            @alias plus = nudged.plus
            @alias minus = nudged.minus
            @alias plus_capture = nudged.plus_capture
            @alias minus_capture = nudged.minus_capture
            @context c1 = forward()
            plus.nudged_beta = clamping_beta
            plus()
            plus_capture(isinggraph = plus.dynamics.model)
            minus.nudged_beta = @transform(x -> -x, clamping_beta)
            minus()
            minus_capture(isinggraph = minus.dynamics.model)
            IsingLearning.set_clamping_beta!(c1.dynamics.model, zero(beta))
            IsingLearning.contrastive_gradient(c1.dynamics.model, plus_capture.captured, minus_capture.captured, clamping_beta, buffers = buffers)
        end
    elseif config.gradient_mode == :plus_free
        Processes.@CompositeAlgorithm begin
            @input clamping_beta = beta
            @state buffers
            @alias plus = nudged.plus
            @alias plus_capture = nudged.plus_capture
            @alias minus_capture = nudged.minus_capture
            @context c1 = forward()
            plus.nudged_beta = clamping_beta
            plus()
            plus_capture(isinggraph = plus.dynamics.model)
            minus_capture(isinggraph = plus.dynamics.model)
            IsingLearning.set_clamping_beta!(c1.dynamics.model, zero(beta))
            edge_difference_gradient!(c1.dynamics.model, plus_capture.captured, c1.equilibrium_state, buffers)
        end
    elseif config.gradient_mode == :free_plus
        Processes.@CompositeAlgorithm begin
            @input clamping_beta = beta
            @state buffers
            @alias plus = nudged.plus
            @alias plus_capture = nudged.plus_capture
            @alias minus_capture = nudged.minus_capture
            @context c1 = forward()
            plus.nudged_beta = clamping_beta
            plus()
            plus_capture(isinggraph = plus.dynamics.model)
            minus_capture(isinggraph = plus.dynamics.model)
            IsingLearning.set_clamping_beta!(c1.dynamics.model, zero(beta))
            edge_difference_gradient!(c1.dynamics.model, c1.equilibrium_state, plus_capture.captured, buffers)
        end
    else
        throw(ArgumentError("EDGE_TWOOUT_GRADIENT_MODE must be symmetric, plus_free, or free_plus; got $(config.gradient_mode)"))
    end
    return (; algorithm = final, plus_capture = nudged.plus_capture, minus_capture = nudged.minus_capture, dynamics = forward.dynamics)
end

"""Create a reusable training process for the zero-init edge composite."""
function edge_worker_process(layer, worker_graph, config::EdgeTwoOutConfig)
    algo = Processes.resolve(edge_forward_and_nudged(layer, config).algorithm)
    buffers = IsingLearning.gradient_buffer(worker_graph)
    return Processes.Process(
        algo,
        Processes.Init(:_state;
            x = zeros(eltype(worker_graph), length(layer.input_layer)),
            y = zeros(eltype(worker_graph), length(layer.output_layer)),
            buffers = buffers,
            equilibrium_state = copy(II.state(worker_graph)),
        ),
        Processes.Init(:dynamics, model = worker_graph),
        Processes.Init(:plus_capture, state = worker_graph),
        Processes.Init(:minus_capture, state = worker_graph);
        repeat = 1,
    )
end

"""Create a reusable validation process for zero-init free relaxation."""
function edge_validation_process(layer, worker_graph, config::EdgeTwoOutConfig)
    algo = Processes.resolve(edge_forward(layer, config).algorithm)
    return Processes.Process(
        algo,
        Processes.Init(:_state;
            x = zeros(eltype(worker_graph), length(layer.input_layer)),
            equilibrium_state = copy(II.state(worker_graph)),
        ),
        Processes.Init(:dynamics, model = worker_graph);
        repeat = 1,
    )
end

"""Initialize the existing trainer container with edge-specific zero-init workers."""
function edge_twoout_trainer(graph, layer, config::EdgeTwoOutConfig)
    params = IsingLearning.read_graph_params(graph)
    optimiser = Optimisers.Adam(config.lr)
    opt_state = Optimisers.setup(optimiser, params)
    template_graph = IsingLearning._worker_graph(graph, params)
    template_worker = edge_worker_process(layer, template_graph, config)
    workers = [
        idx == 1 ? template_worker :
        Processes.copyprocess(template_worker; context = deepcopy(Processes.context(template_worker)))
        for idx in 1:config.workers
    ]
    worker_graphs = [Processes.context(worker).dynamics.model for worker in workers]
    validation_graph = IsingLearning._worker_graph(graph, params)
    validation_worker = edge_validation_process(layer, validation_graph, config)
    return IsingLearning.MNISTThreadedTrainer(
        layer,
        graph,
        params,
        opt_state,
        worker_graphs,
        workers,
        validation_graph,
        validation_worker,
        optimiser,
    )
end

"""Repeat the four XOR cases to average stochastic EqProp gradients."""
function edge_repeated_batch(x, y, minit)
    xbatch = Matrix{FT}(undef, size(x, 1), size(x, 2) * minit)
    ybatch = Matrix{FT}(undef, size(y, 1), size(y, 2) * minit)
    col = 1
    for _ in 1:minit, sample in axes(x, 2)
        xbatch[:, col] .= view(x, :, sample)
        ybatch[:, col] .= view(y, :, sample)
        col += 1
    end
    return xbatch, ybatch
end

"""Evaluate repeated validation starts with the existing validation worker."""
function edge_evaluate!(trainer, x, y, config::EdgeTwoOutConfig)
    xeval, yeval = edge_repeated_batch(x, y, config.eval_repeats)
    II.temp!(trainer.validation_graph, config.validation_temp)
    nsamples = size(xeval, 2)
    ncorrect = 0
    total_squared_error = zero(eltype(trainer.params.w))
    for sample_idx in 1:nsamples
        worker = trainer.validation_worker
        IsingLearning._write_input!(worker, view(xeval, :, sample_idx))
        Processes.reset!(worker)
        run(worker)
        wait(worker)
        close(worker)

        output = @view Processes.context(worker)._state.equilibrium_state[trainer.layer.output_layer]
        target = view(yeval, :, sample_idx)
        total_squared_error += sum(abs2, output .- target)
        out0 = mean(@view output[1:config.output_repeats])
        out1 = mean(@view output[(config.output_repeats + 1):(2 * config.output_repeats)])
        tgt0 = mean(@view target[1:config.output_repeats])
        tgt1 = mean(@view target[(config.output_repeats + 1):(2 * config.output_repeats)])
        ncorrect += (out1 > out0) == (tgt1 > tgt0)
    end
    return (;
        mse = total_squared_error / (nsamples * size(y, 1)),
        acc = ncorrect / nsamples,
    )
end

"""Run one minibatch while passing the clamping strength as a runtime input."""
function edge_run_minibatch!(trainer, xbatch, ybatch, batch_gradient, clamping_beta)
    IsingLearning._reset_batch_buffers!(trainer)

    batchsize = size(xbatch, 2)
    workers = trainer.workers
    for sample_idx in 1:batchsize
        worker_idx = nothing
        while true
            worker_idx = findfirst(worker -> isnothing(worker.task) || Processes.isdone(worker), workers)
            isnothing(worker_idx) || break
            yield()
        end

        worker = workers[something(worker_idx)]
        Processes.isdone(worker) && close(worker)
        IsingLearning._write_example!(worker, view(xbatch, :, sample_idx), view(ybatch, :, sample_idx))
        Processes.reset!(worker)
        run(worker; clamping_beta = clamping_beta)
    end

    for worker in workers
        if !isnothing(worker.task)
            wait(worker)
            close(worker)
        end
    end

    IsingLearning._collect_batch_gradient!(trainer, batch_gradient, batchsize)
    trainer.opt_state, trainer.params = Optimisers.update(trainer.opt_state, trainer.params, batch_gradient)
    IsingLearning._broadcast_params!(trainer)
    return nothing
end

"""Write dictionary rows to CSV."""
function edge_write_csv(path, rows)
    isempty(rows) && return path
    headers = sort!(collect(keys(first(rows))))
    open(path, "w") do io
        println(io, join(headers, ","))
        for row in rows
            println(io, join((row[h] for h in headers), ","))
        end
    end
    return path
end

"""Plot MSE and accuracy over training."""
function edge_plot(path, rows)
    fig = Figure(size = (900, 520))
    epochs = [r["epoch"] for r in rows]
    ax1 = Axis(fig[1, 1], title = "edge 2->8x8->2 MSE", xlabel = "epoch", ylabel = "MSE")
    ax2 = Axis(fig[1, 2], title = "accuracy", xlabel = "epoch", ylabel = "accuracy")
    lines!(ax1, epochs, [r["mse"] for r in rows])
    lines!(ax2, epochs, [r["accuracy"] for r in rows])
    save(path, fig)
    return path
end

"""Train one `2 -> 8x8 -> 2` edge-connected XOR model."""
function train_edge_twoout(config::EdgeTwoOutConfig; outdir)
    mkpath(outdir)
    graph = edge_twoout_graph(config)
    layer = edge_twoout_layer(graph, config)
    trainer = edge_twoout_trainer(graph, layer, config)
    x, y = edge_twoout_dataset(config)
    xbatch, ybatch = edge_repeated_batch(x, y, config.minit)
    batch_gradient = IsingLearning.gradient_buffer(trainer.prototype_graph)
    rows = Dict{String,Any}[]

    metrics = edge_evaluate!(trainer, x, y, config)
    best = (mse = metrics.mse, acc = metrics.acc, epoch = 0, params = deepcopy(trainer.params))
    push!(rows, Dict("epoch" => 0, "mse" => metrics.mse, "accuracy" => metrics.acc, "grad_norm" => zero(FT)))
    println("epoch=0 mse=", round(metrics.mse, digits = 6), " acc=", metrics.acc)

    for epoch in 1:config.epochs
        edge_run_minibatch!(trainer, xbatch, ybatch, batch_gradient, config.β)
        if config.weight_decay > 0
            trainer.params.w .*= (one(FT) - config.lr * config.weight_decay)
            trainer.params.b .*= (one(FT) - config.lr * config.weight_decay)
            IsingLearning._broadcast_params!(trainer)
        end
        if epoch == 1 || epoch % config.log_every == 0 || epoch == config.epochs
            metrics = edge_evaluate!(trainer, x, y, config)
            grad_norm = sqrt(sum(abs2, batch_gradient.w) + sum(abs2, batch_gradient.b))
            push!(rows, Dict("epoch" => epoch, "mse" => metrics.mse, "accuracy" => metrics.acc, "grad_norm" => grad_norm))
            if metrics.acc > best.acc || (metrics.acc == best.acc && metrics.mse < best.mse)
                best = (mse = metrics.mse, acc = metrics.acc, epoch = epoch, params = deepcopy(trainer.params))
            end
            println("epoch=", epoch, " mse=", round(metrics.mse, digits = 6),
                " acc=", metrics.acc, " grad=", round(grad_norm, digits = 4))
        end
    end

    edge_write_csv(joinpath(outdir, "metrics.csv"), rows)
    edge_plot(joinpath(outdir, "progress.png"), rows)
    trainer.params = best.params
    IsingLearning._broadcast_params!(trainer)
    II.save_isinggraph(joinpath(outdir, "best_graph.jld2"), deepcopy(trainer.prototype_graph))
    close_trainer!(trainer)
    return (; best, rows, outdir)
end

"""Run the default edge-connected two-output experiment."""
function main()
    config = EdgeTwoOutConfig()
    outdir = joinpath(@__DIR__, "runs", "edge_2_8x8_2_" * Dates.format(now(), "yyyymmdd_HHMMSS"))
    result = train_edge_twoout(config; outdir)
    println("RESULT best_mse=", result.best.mse, " best_acc=", result.best.acc,
        " best_epoch=", result.best.epoch, " outdir=", outdir)
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
