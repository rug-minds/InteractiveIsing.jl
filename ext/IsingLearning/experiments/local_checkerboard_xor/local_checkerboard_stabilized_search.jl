include("local_checkerboard_xor.jl")

"""
    StabilizedSearchConfig

Experiment-local wrapper around `LocalCheckerboardConfig`.

This file deliberately does not change the toolbox or the main checkerboard
experiment. It tries a few stabilization ideas around the already-promising
2x2 local checkerboard Metropolis runs:

- remove the temperature-annealing sampler wrapper for Metropolis;
- optionally clip weights/biases after every optimizer update;
- optionally rescale all weights to keep the maximum local field in a target
  range, so the chosen temperature remains meaningful;
- save full run artifacts only if the run reaches a useful score.
"""
Base.@kwdef struct StabilizedSearchConfig
    config::LocalCheckerboardConfig
    save_threshold::FT = 0.5
    max_abs_weight::Union{Nothing,FT} = nothing
    max_abs_bias::Union{Nothing,FT} = nothing
    zero_bias::Bool = false
    target_max_local_energy::Union{Nothing,FT} = nothing
    gradient_multiplier::FT = one(FT)
    initial_false_output_bias::Union{Nothing,FT} = nothing
    input_default_bias::Union{Nothing,FT} = nothing
    fix_input_default_bias::Bool = false
    full_bipolar_input::Bool = false
    input_kick::Bool = false
    freeze_inactive_input::Bool = false
    case_repeats::NTuple{4,Int} = (1, 1, 1, 1)
    notes::String = ""
end

function fresh_checker_dynamics(config::LocalCheckerboardConfig)
    if config.dynamics_mode === :metropolis
        return II.Metropolis()
    elseif config.dynamics_mode === :langevin
        return II.BlockLangevin(
            stepsize = config.stepsize,
            adjusted = false,
            block_size = config.block_size,
            group_steps = 1,
        )
    elseif config.dynamics_mode === :local_langevin
        return II.LocalLangevin(
            stepsize = config.stepsize,
            adjusted = false,
            order = :random,
            group_steps = 1,
        )
    elseif config.dynamics_mode === :global_langevin
        return II.GlobalLangevin(
            stepsize = config.stepsize,
            adjusted = false,
            group_steps = 1,
        )
    else
        throw(ArgumentError("unsupported dynamics_mode $(config.dynamics_mode)"))
    end
end

function stable_prepare_free_state!(g, config::LocalCheckerboardConfig)
    if config.init_mode === :persistent
        if !get(g.addons, :stable_persistent_initialized, false)
            II.state(g) .= II.initRandomState(g)
            g.addons[:stable_persistent_initialized] = true
        end
        set_input_case!(g.index_set, 1)
    elseif config.init_mode === :minus
        fill!(II.state(g), -one(eltype(II.state(g))))
        set_input_case!(g.index_set, 1)
    else
        checker_initstate!(g, config)
    end
    return g
end

"""
    stable_apply_input!(g, x, config, search)

Apply the input convention selected by an experiment wrapper.

With `full_bipolar_input=true`, the two XOR bits are embedded directly into the
two checkerboard masks: bit `0` writes `-1`, bit `1` writes `+1`, and the whole
input code is frozen. This is the standard fixed-input EqProp condition for the
checkerboard code.

The older `input_kick`/`freeze_inactive_input` flags are kept only for old
experiments that intentionally used the "missing bit is free" convention.
"""
function stable_apply_input!(g, x, config::LocalCheckerboardConfig, search::StabilizedSearchConfig)
    if search.full_bipolar_input
        positions = checker_code_positions(config.side, config.code_side, config.code_stride, config.code_offset)
        masks = checker_masks(positions)
        a_idxs = global_idxs_for_positions(g[1], masks.a)
        b_idxs = global_idxs_for_positions(g[1], masks.b)
        aval = x[1] > 0 ? one(eltype(II.state(g))) : -one(eltype(II.state(g)))
        bval = x[2] > 0 ? one(eltype(II.state(g))) : -one(eltype(II.state(g)))
        s = II.state(g)
        @inbounds begin
            for idx in a_idxs
                s[idx] = aval
            end
            for idx in b_idxs
                s[idx] = bval
            end
        end
        code_set = Set(unique(vcat(a_idxs, b_idxs)))
        g.index_set.active[] = [idx for idx in g.index_set.all_active if !(idx in code_set)]
        g.index_set.changed[] = true
        return g
    end

    apply_checker_input!(g, x, config)
    (search.input_kick || search.freeze_inactive_input) || return g

    positions = checker_code_positions(config.side, config.code_side, config.code_stride, config.code_offset)
    masks = checker_masks(positions)
    a_idxs = global_idxs_for_positions(g[1], masks.a)
    b_idxs = global_idxs_for_positions(g[1], masks.b)
    active_a = x[1] > 0
    active_b = x[2] > 0
    s = II.state(g)
    @inbounds begin
        if !active_a
            for idx in a_idxs
                s[idx] = -one(eltype(s))
            end
        end
        if !active_b
            for idx in b_idxs
                s[idx] = -one(eltype(s))
            end
        end
    end
    if search.freeze_inactive_input
        code_idxs = unique(vcat(a_idxs, b_idxs))
        code_set = Set(code_idxs)
        g.index_set.active[] = [idx for idx in g.index_set.all_active if !(idx in code_set)]
        g.index_set.changed[] = true
    end
    return g
end

"""
    StableForwardDynamics(layer, config)

Forward relaxation used by this search. It is intentionally the same as
`CheckerForwardDynamics`, except that it does not wrap the sampler in
`TemperatureAnnealedSampler`. This is important for the discrete Metropolis
control path, where the wrapper caused a Processes context merge failure.
"""
function StableForwardDynamics(layer, config::LocalCheckerboardConfig, search::StabilizedSearchConfig = StabilizedSearchConfig(config = config))
    relaxation_steps = layer.free_relaxation_steps
    n_units = layer.nunits
    dynamics_algorithm = fresh_checker_dynamics(config)

    forward = @Routine begin
        @alias dynamics = dynamics_algorithm
        @state equilibrium_state = zeros(n_units)
        @state x

        stable_prepare_free_state!(dynamics.model, config)
        stable_apply_input!(dynamics.model, x, config, search)
        model = @repeat relaxation_steps dynamics()
        IsingLearning.copyvector!(equilibrium_state, @transform(m -> II.state(m), model))
    end
    return (; algorithm = forward, dynamics = forward.dynamics)
end

"""
    StableNudgedDynamics(layer, config)

    Plus/minus nudged phases without annealing wrappers.
"""
function StableNudgedDynamics(layer, config::LocalCheckerboardConfig, search::StabilizedSearchConfig = StabilizedSearchConfig(config = config))
    beta = layer.β
    relaxation_steps = layer.nudged_relaxation_steps
    plus_capture = IsingLearning.Capturer()
    minus_capture = IsingLearning.Capturer()
    plus_dynamics_algorithm = fresh_checker_dynamics(config)
    minus_dynamics_algorithm = fresh_checker_dynamics(config)

    plus = @Routine begin
        @state equilibrium_state
        @state y
        @state x
        @alias dynamics = plus_dynamics_algorithm
        @alias plus_capture = plus_capture

        IsingLearning.setgraph!(isinggraph = dynamics.model, target = equilibrium_state)
        stable_apply_input!(dynamics.model, x, config, search)
        checker_apply_targets!(dynamics.model, y)
        checker_set_clamping_beta!(dynamics.model, beta)
        model = @repeat relaxation_steps dynamics()
        plus_capture(isinggraph = model)
    end

    minus = @Routine begin
        @state equilibrium_state
        @state y
        @state x
        @alias dynamics = minus_dynamics_algorithm
        @alias minus_capture = minus_capture

        IsingLearning.setgraph!(isinggraph = dynamics.model, target = equilibrium_state)
        stable_apply_input!(dynamics.model, x, config, search)
        checker_apply_targets!(dynamics.model, y)
        checker_set_clamping_beta!(dynamics.model, -beta)
        model = @repeat relaxation_steps dynamics()
        minus_capture(isinggraph = model)
    end

    final = @CompositeAlgorithm begin
        @state buffers
        @context c1 = plus()
        @context c2 = minus()
    end
    return (; algorithm = final, plus_capture, minus_capture, dynamics = plus.dynamics)
end

function StableForwardAndNudged(layer, config::LocalCheckerboardConfig, search::StabilizedSearchConfig = StabilizedSearchConfig(config = config))
    forward = StableForwardDynamics(layer, config, search).algorithm
    nudged = StableNudgedDynamics(layer, config, search)
    beta = layer.β
    final = @CompositeAlgorithm begin
        @state buffers
        @context c1 = forward()
        @context c2 = nudged.algorithm()
        checker_set_clamping_beta!(c1.dynamics.model, zero(beta))
        IsingLearning.contrastive_gradient(c1.dynamics.model, c2.plus_capture.captured, c2.minus_capture.captured, beta, buffers = buffers)
    end
    return (; algorithm = final, plus_capture = nudged.plus_capture, minus_capture = nudged.minus_capture, dynamics = forward.dynamics)
end

function stable_checker_worker_process(layer, graph, search::StabilizedSearchConfig)
    config = search.config
    algo = Processes.resolve(StableForwardAndNudged(layer, config, search).algorithm)
    buffers = IsingLearning.gradient_buffer(graph)
    return Process(
        algo,
        Init(:_state;
            x = zeros(FT, 2),
            y = zeros(FT, target_dim(config)),
            buffers = buffers,
            equilibrium_state = copy(II.state(graph)),
        ),
        dynamics_input(:dynamics, graph, config.base_seed),
        Init(:plus_capture, state = graph),
        Init(:minus_capture, state = graph);
        repeat = 1,
    )
end

function stable_checker_validation_process(layer, graph, search::StabilizedSearchConfig)
    config = search.config
    algo = Processes.resolve(StableForwardDynamics(layer, config, search).algorithm)
    return Process(
        algo,
        Init(:_state;
            x = zeros(FT, 2),
            equilibrium_state = copy(II.state(graph)),
        ),
        dynamics_input(:dynamics, graph, config.base_seed + 50_000);
        repeat = 1,
    )
end

function init_stable_checker_trainer(layer, search::StabilizedSearchConfig; graph = layer.model_graph, optimiser = Optimisers.Adam(search.config.lr))
    config = search.config
    params = IsingLearning.read_graph_params(graph)
    opt_state = Optimisers.setup(optimiser, params)
    workers = Process[]
    worker_graphs = typeof(graph)[]
    for _ in 1:config.workers
        wg = IsingLearning._worker_graph(graph, params)
        II.temp!(wg, effective_temp(wg, config))
        push!(worker_graphs, wg)
        push!(workers, stable_checker_worker_process(layer, wg, search))
    end
    validation_graph = IsingLearning._worker_graph(graph, params)
    II.temp!(validation_graph, effective_temp(validation_graph, config))
    validation_worker = stable_checker_validation_process(layer, validation_graph, search)
    return CheckerTrainer(layer, graph, params, opt_state, worker_graphs, workers, validation_graph, validation_worker, optimiser)
end

function clip_graph_parameters!(graph, search::StabilizedSearchConfig)
    A = II.adj(graph).sp
    vals = SparseArrays.getnzval(A)
    if !isnothing(search.max_abs_weight)
        clamp!(vals, -search.max_abs_weight, search.max_abs_weight)
    end
    if !isnothing(search.target_max_local_energy)
        energy = max_local_interaction_energy(graph)
        if isfinite(energy) && energy > eps(FT)
            vals .*= search.target_max_local_energy / energy
        end
    end
    symmetrize_adjacency!(graph)

    b = II.getparam(graph.hamiltonian, II.MagField, :b)
    if search.zero_bias
        fill!(b, zero(eltype(b)))
    elseif !isnothing(search.max_abs_bias)
        clamp!(b, -search.max_abs_bias, search.max_abs_bias)
    end
    return graph
end

function project_trainer_params!(trainer::CheckerTrainer, search::StabilizedSearchConfig)
    isnothing(search.max_abs_weight) &&
        isnothing(search.max_abs_bias) &&
        isnothing(search.target_max_local_energy) &&
        !search.zero_bias && return trainer

    IsingLearning.sync_graph_params!(trainer.prototype_graph, trainer.params)
    clip_graph_parameters!(trainer.prototype_graph, search)
    trainer.params = IsingLearning.read_graph_params(trainer.prototype_graph)
    _broadcast_params!(trainer)
    return trainer
end

function apply_initial_false_output_bias!(graph, config::LocalCheckerboardConfig, search::StabilizedSearchConfig)
    isnothing(search.initial_false_output_bias) && return graph
    output_idxs = get(graph, :checker_output_idxs, nothing)
    isnothing(output_idxs) && error("graph is missing :checker_output_idxs addon")
    b = II.getparam(graph.hamiltonian, II.MagField, :b)
    false_pattern = checker_output_target_vector(config, -one(FT))
    @inbounds for (idx, target) in zip(output_idxs, false_pattern)
        b[idx] += search.initial_false_output_bias * target
    end
    return graph
end

function checker_input_code_idxs(graph, config::LocalCheckerboardConfig)
    positions = checker_code_positions(config.side, config.code_side, config.code_stride, config.code_offset)
    masks = checker_masks(positions)
    return unique(vcat(
        global_idxs_for_positions(graph[1], masks.a),
        global_idxs_for_positions(graph[1], masks.b),
    ))
end

"""
    apply_input_default_bias!(graph, config, search)

Apply an experiment-local physical default for zero input bits. The XOR encoding
still only freezes active bits to `+1`; inactive bits are not clamped. A
negative magnetic field on the input-code sites makes those inactive bits prefer
`-1` during free relaxation.
"""
function apply_input_default_bias!(graph, config::LocalCheckerboardConfig, search::StabilizedSearchConfig)
    isnothing(search.input_default_bias) && return graph
    input_idxs = checker_input_code_idxs(graph, config)
    b = II.getparam(graph.hamiltonian, II.MagField, :b)
    @inbounds for idx in input_idxs
        b[idx] = -abs(FT(search.input_default_bias))
    end
    graph.addons[:checker_input_default_idxs] = input_idxs
    graph.addons[:checker_input_default_bias] = -abs(FT(search.input_default_bias))
    return graph
end

function project_input_default_bias!(trainer::CheckerTrainer, search::StabilizedSearchConfig)
    (isnothing(search.input_default_bias) || !search.fix_input_default_bias) && return trainer
    input_idxs = get(trainer.prototype_graph.addons, :checker_input_default_idxs, nothing)
    isnothing(input_idxs) && return trainer
    fixed_bias = get(trainer.prototype_graph.addons, :checker_input_default_bias, -abs(FT(search.input_default_bias)))
    @inbounds for idx in input_idxs
        trainer.params.b[idx] = fixed_bias
    end
    return trainer
end

function _set_symmetric_weight!(graph, i::Integer, j::Integer, value)
    A = II.adj(graph).sp
    A[Int(i), Int(j)] = FT(value)
    A[Int(j), Int(i)] = FT(value)
    return graph
end

function _zero_between_layers!(graph, layer_a, layer_b)
    for i in II.graphidxs(layer_a), j in II.graphidxs(layer_b)
        _set_symmetric_weight!(graph, i, j, zero(FT))
    end
    return graph
end

function train_epoch_stable!(trainer::CheckerTrainer, x, y, batch_gradient, epoch::Integer, search::StabilizedSearchConfig)
    config = search.config
    IsingLearning.zero_buffer!(batch_gradient)
    nsamples = size(x, 2)
    responses = FT[]
    task_batch = Process[]
    seed_base = config.base_seed + epoch * 100_000
    job = 0
    for sample_idx in 1:nsamples
        for init_idx in 1:config.minit
            job += 1
            worker = trainer.workers[mod1(job, length(trainer.workers))]
            start_worker!(worker, @view(x[:, sample_idx]), @view(y[:, sample_idx]); seed = seed_base + 257 * sample_idx + init_idx)
            push!(task_batch, worker)
            if length(task_batch) == length(trainer.workers)
                for task_worker in task_batch
                    finish_worker!(task_worker, batch_gradient, responses)
                end
                empty!(task_batch)
            end
        end
    end
    for task_worker in task_batch
        finish_worker!(task_worker, batch_gradient, responses)
    end
    scale = search.gradient_multiplier * inv(FT(2) * FT(config.β) * FT(nsamples * config.minit))
    IsingLearning.scale_buffer!(batch_gradient, scale)
    add_weight_decay!(batch_gradient, trainer.params, config.weight_decay)
    if !isnothing(search.input_default_bias) && search.fix_input_default_bias
        input_idxs = get(trainer.prototype_graph.addons, :checker_input_default_idxs, nothing)
        if !isnothing(input_idxs)
            @inbounds for idx in input_idxs
                batch_gradient.b[idx] = zero(eltype(batch_gradient.b))
            end
        end
    end
    clip_gradient!(batch_gradient, config.grad_clip)

    trainer.opt_state, trainer.params = Optimisers.update(trainer.opt_state, trainer.params, batch_gradient)
    project_input_default_bias!(trainer, search)
    _broadcast_params!(trainer)
    grad_norm = sqrt(sum(abs2, batch_gradient.w) + sum(abs2, batch_gradient.b))
    grad = (; grad_norm, response_norm = mean(responses))
    project_trainer_params!(trainer, search)
    return grad
end

function run_stabilized_config(search::StabilizedSearchConfig, outdir)
    config = search.config
    mkpath(outdir)
    graph = checkerboard_graph(config)
    apply_input_default_bias!(graph, config, search)
    apply_initial_false_output_bias!(graph, config, search)
    clip_graph_parameters!(graph, search)
    layer = checkerboard_layer(graph, config)
    trainer = init_stable_checker_trainer(layer, search; graph, optimiser = Optimisers.Adam(config.lr))
    x, y = xor_inputs_targets(config)
    if search.case_repeats != (1, 1, 1, 1)
        cols = reduce(vcat, (fill(i, search.case_repeats[i]) for i in 1:4))
        x_train = x[:, cols]
        y_train = y[:, cols]
    else
        x_train = x
        y_train = y
    end
    batch_gradient = IsingLearning.gradient_buffer(trainer.prototype_graph)
    initial_params = deepcopy(trainer.params)
    rows = Dict{String,Any}[]
    best_params = deepcopy(trainer.params)
    best_mse = Inf
    best_acc = -Inf
    zero_grad = (; grad_norm = zero(FT), response_norm = zero(FT))

    eval_seed = config.base_seed + 50_000_000
    metrics = evaluate_checker!(trainer, x, y, config; seed_offset = eval_seed)
    push!(rows, metric_row(config.name, 0, metrics, zero_grad, trainer, initial_params))
    print_metrics(0, metrics, zero_grad)
    if metrics.accuracy > best_acc || (metrics.accuracy == best_acc && metrics.mse < best_mse)
        best_acc, best_mse, best_params = metrics.accuracy, metrics.mse, deepcopy(trainer.params)
    end

    for epoch in 1:config.epochs
        grad = train_epoch_stable!(trainer, x_train, y_train, batch_gradient, epoch, search)
        if epoch == 1 || epoch % config.log_every == 0 || epoch == config.epochs
            metrics = evaluate_checker!(trainer, x, y, config; seed_offset = eval_seed)
            push!(rows, metric_row(config.name, epoch, metrics, grad, trainer, initial_params))
            print_metrics(epoch, metrics, grad)
            if metrics.accuracy > best_acc || (metrics.accuracy == best_acc && metrics.mse < best_mse)
                best_acc, best_mse, best_params = metrics.accuracy, metrics.mse, deepcopy(trainer.params)
            end
        end
    end

    trainer.params = best_params
    _broadcast_params!(trainer)
    graph_path = nothing
    svg_path = nothing
    if best_mse <= search.save_threshold
        strip_weight_generators!(trainer.prototype_graph)
        graph_path = II.save_isinggraph(joinpath(outdir, "$(config.name)_best_graph.jld2"), trainer.prototype_graph)
        svg_path = write_parameter_svg(joinpath(outdir, "$(config.name)_parameters.svg"), trainer.prototype_graph, config)
    end
    close_checker_trainer!(trainer)
    return (; rows, graph_path, svg_path, best_mse, best_acc, notes = search.notes)
end

function write_stabilized_readme(path, searches, results, csv_path, png_path)
    open(path, "w") do io
        println(io, "# Local Checkerboard Stabilized Search")
        println(io)
        println(io, "This run is isolated from the toolbox code. It reuses the checkerboard graph/trainer helpers but replaces the annealed sampler wrapper with no-anneal Processes composites for this search.")
        println(io)
        println(io, "## Brainstormed Fixes Tested")
        println(io)
        println(io, "- Metropolis no-anneal control, because the previous best local checkerboard runs were discrete Metropolis and the annealing wrapper currently breaks that path.")
        println(io, "- Post-update weight clipping, to keep the system in a temperature/coupling regime where states are neither frozen nor noise dominated.")
        println(io, "- Optional local-field normalization, to couple the maximum local interaction scale to the chosen temperature.")
        println(io, "- Bias suppression/clipping probes, because strong biases can solve single cases while hurting XOR symmetry.")
        println(io)
        println(io, "## Results")
        println(io)
        println(io, "| Config | Best MSE | Best Acc | Saved | Notes |")
        println(io, "|---|---:|---:|---|---|")
        for (search, result) in zip(searches, results)
            saved = isnothing(result.graph_path) ? "no" : "yes"
            println(io, "| `$(search.config.name)` | $(round(result.best_mse, digits=6)) | $(round(result.best_acc, digits=3)) | $saved | $(search.notes) |")
        end
        println(io)
        println(io, "Metrics CSV: `$(basename(csv_path))`")
        println(io)
        println(io, "Progress PNG: `$(basename(png_path))`")
    end
    return path
end

function stabilized_searches()
    epochs = parse(Int, get(ENV, "ISING_STABLE_XOR_EPOCHS", "2500"))
    log_every = parse(Int, get(ENV, "ISING_STABLE_XOR_LOG_EVERY", "250"))
    workers = parse(Int, get(ENV, "ISING_STABLE_XOR_THREADS", string(max(1, min(Threads.nthreads(), 8)))))
    minit = parse(Int, get(ENV, "ISING_STABLE_XOR_MINIT", "8"))
    eval_repeats = parse(Int, get(ENV, "ISING_STABLE_XOR_EVAL_REPEATS", "32"))
    free_relaxation = parse(Int, get(ENV, "ISING_STABLE_XOR_FREE_RELAXATION", "150"))
    nudged_relaxation = parse(Int, get(ENV, "ISING_STABLE_XOR_NUDGED_RELAXATION", "150"))
    quick = parse(Bool, get(ENV, "ISING_STABLE_XOR_QUICK", "false"))
    common = (;
        side = 2,
        hidden_side = 2,
        code_side = 2,
        code_stride = 1,
        epochs = quick ? min(epochs, 2) : epochs,
        log_every = quick ? 1 : log_every,
        minit = quick ? 1 : minit,
        eval_repeats = quick ? 2 : eval_repeats,
        workers,
        free_relaxation = quick ? 5 : free_relaxation,
        nudged_relaxation = quick ? 5 : nudged_relaxation,
        β = parse(FT, get(ENV, "ISING_STABLE_XOR_BETA", "0.2")),
        lr = parse(FT, get(ENV, "ISING_STABLE_XOR_LR", "0.005")),
        weight_decay = parse(FT, get(ENV, "ISING_STABLE_XOR_WEIGHT_DECAY", "0.0")),
        grad_clip = FT(20),
        inter_radius = sqrt(2.0) + 1e-6,
        internal_nn = 1,
        input_internal_scale = FT(0.02),
        hidden_internal_scale = FT(0.02),
        output_internal_scale = FT(0.02),
        bias_scale = FT(0.02),
        weight_seed = 2,
        internal_seed = 3,
        bias_seed = 11,
        base_seed = 91000,
        init_mode = :random,
        dynamics_mode = :metropolis,
        output_clamp_mode = :readout,
        doublewell_barrier = FT(0),
    )
    searches = [
        StabilizedSearchConfig(
            config = LocalCheckerboardConfig(;
                name = "metro_T0015_J005_long",
                temp = FT(0.015),
                inter_weight_scale = FT(0.05),
                state_mode = :discrete,
                common...,
            ),
            notes = "closest reconstruction of the old sub-0.2 Metropolis run: small initial J, no projection",
        ),
        StabilizedSearchConfig(
            config = LocalCheckerboardConfig(;
                name = "metro_continuous_T0015_J005_long",
                temp = FT(0.015),
                inter_weight_scale = FT(0.05),
                state_mode = :continuous,
                common...,
            ),
            notes = "continuous-state Metropolis reconstruction; the old reference run likely used this default state mode",
        ),
        StabilizedSearchConfig(
            config = LocalCheckerboardConfig(;
                name = "metro_T0015_J012_flipgrad",
                temp = FT(0.015),
                inter_weight_scale = FT(0.12),
                state_mode = :discrete,
                common...,
            ),
            gradient_multiplier = -one(FT),
            notes = "opposite gradient direction probe; isolated check because current direction damps margins in this setup",
        ),
        StabilizedSearchConfig(
            config = LocalCheckerboardConfig(;
                name = "metro_hot_T005_J012_strongbeta",
                temp = FT(0.05),
                inter_weight_scale = FT(0.12),
                state_mode = :discrete,
                common...,
            ),
            max_abs_weight = FT(1.5),
            max_abs_bias = FT(0.75),
            notes = "hotter Metropolis plus stronger beta probe; intended to keep plus/minus nudged states separated",
        ),
        StabilizedSearchConfig(
            config = LocalCheckerboardConfig(;
                common...,
                name = "metro_oldrecipe_random_init",
                temp = FT(0.015),
                temp_is_factor = true,
                inter_weight_scale = FT(0.25),
                input_internal_scale = FT(0.1),
                hidden_internal_scale = FT(0.1),
                output_internal_scale = FT(0.1),
                weight_seed = 13,
                internal_seed = 14,
                bias_seed = 22,
                base_seed = 93016,
                state_mode = :discrete,
            ),
            notes = "old successful hyperparameters except random init instead of the old invalid zero init",
        ),
        StabilizedSearchConfig(
            config = LocalCheckerboardConfig(;
                common...,
                name = "metro_oldrecipe_persistent",
                temp = FT(0.015),
                temp_is_factor = true,
                inter_weight_scale = FT(0.25),
                input_internal_scale = FT(0.1),
                hidden_internal_scale = FT(0.1),
                output_internal_scale = FT(0.1),
                weight_seed = 13,
                internal_seed = 14,
                bias_seed = 22,
                base_seed = 93016,
                init_mode = :persistent,
                state_mode = :discrete,
            ),
            notes = "persistent random chains: random first state, then no per-free-phase random reset",
        ),
        StabilizedSearchConfig(
            config = LocalCheckerboardConfig(;
                common...,
                name = "metro_oldrecipe_pattern_clamp",
                temp = FT(0.015),
                temp_is_factor = true,
                inter_weight_scale = FT(0.25),
                input_internal_scale = FT(0.1),
                hidden_internal_scale = FT(0.1),
                output_internal_scale = FT(0.1),
                weight_seed = 13,
                internal_seed = 14,
                bias_seed = 22,
                base_seed = 93016,
                state_mode = :discrete,
                output_clamp_mode = :pattern,
            ),
            notes = "direct physical output-pattern clamping instead of scalar readout clamping",
        ),
        StabilizedSearchConfig(
            config = LocalCheckerboardConfig(;
                common...,
                name = "metro_oldrecipe_false_output_prior",
                temp = FT(0.015),
                temp_is_factor = true,
                inter_weight_scale = FT(0.25),
                input_internal_scale = FT(0.1),
                hidden_internal_scale = FT(0.1),
                output_internal_scale = FT(0.1),
                weight_seed = 13,
                internal_seed = 14,
                bias_seed = 22,
                base_seed = 93016,
                state_mode = :discrete,
            ),
            initial_false_output_bias = FT(0.25),
            notes = "initialize output biases toward the false code so the no-input case has a default basin",
        ),
        StabilizedSearchConfig(
            config = LocalCheckerboardConfig(;
                common...,
                name = "metro_false_prior_pattern_clamp",
                temp = FT(0.015),
                temp_is_factor = true,
                inter_weight_scale = FT(0.25),
                input_internal_scale = FT(0.1),
                hidden_internal_scale = FT(0.1),
                output_internal_scale = FT(0.1),
                weight_seed = 13,
                internal_seed = 14,
                bias_seed = 22,
                base_seed = 93016,
                state_mode = :discrete,
                output_clamp_mode = :pattern,
                β = FT(0.2),
                lr = FT(0.002),
            ),
            initial_false_output_bias = FT(0.25),
            notes = "false output prior plus direct output-pattern clamping",
        ),
        StabilizedSearchConfig(
            config = LocalCheckerboardConfig(;
                common...,
                name = "metro_oldrecipe_input_kick",
                temp = FT(0.015),
                temp_is_factor = true,
                inter_weight_scale = FT(0.25),
                input_internal_scale = FT(0.1),
                hidden_internal_scale = FT(0.1),
                output_internal_scale = FT(0.1),
                weight_seed = 13,
                internal_seed = 14,
                bias_seed = 22,
                base_seed = 93016,
                state_mode = :discrete,
            ),
            input_kick = true,
            notes = "initially write inactive input checkerboard sites to -1, but keep them unfrozen",
        ),
        StabilizedSearchConfig(
            config = LocalCheckerboardConfig(;
                common...,
                name = "metro_oldrecipe_full_bipolar_input",
                temp = FT(0.015),
                temp_is_factor = true,
                inter_weight_scale = FT(0.25),
                input_internal_scale = FT(0.1),
                hidden_internal_scale = FT(0.1),
                output_internal_scale = FT(0.1),
                weight_seed = 13,
                internal_seed = 14,
                bias_seed = 22,
                base_seed = 93016,
                state_mode = :discrete,
            ),
            input_kick = true,
            freeze_inactive_input = true,
            notes = "diagnostic full bipolar input clamp: inactive checkerboard sites are frozen to -1",
        ),
        StabilizedSearchConfig(
            config = LocalCheckerboardConfig(;
                common...,
                name = "metro_full_bipolar_zero_bias",
                temp = FT(0.015),
                temp_is_factor = true,
                inter_weight_scale = FT(0.25),
                input_internal_scale = FT(0.1),
                hidden_internal_scale = FT(0.1),
                output_internal_scale = FT(0.1),
                weight_seed = 13,
                internal_seed = 14,
                bias_seed = 22,
                base_seed = 93016,
                state_mode = :discrete,
            ),
            input_kick = true,
            freeze_inactive_input = true,
            zero_bias = true,
            notes = "full bipolar input with all trainable biases projected to zero after each update",
        ),
        StabilizedSearchConfig(
            config = LocalCheckerboardConfig(;
                common...,
                name = "metro_full_bipolar_pattern_beta05",
                temp = FT(0.015),
                temp_is_factor = true,
                inter_weight_scale = FT(0.25),
                input_internal_scale = FT(0.1),
                hidden_internal_scale = FT(0.1),
                output_internal_scale = FT(0.1),
                weight_seed = 13,
                internal_seed = 14,
                bias_seed = 22,
                base_seed = 93016,
                state_mode = :discrete,
                output_clamp_mode = :pattern,
                β = FT(0.5),
                lr = FT(0.0015),
            ),
            input_kick = true,
            freeze_inactive_input = true,
            notes = "full bipolar input plus direct output-pattern clamping with stronger beta and lower lr",
        ),
        StabilizedSearchConfig(
            config = LocalCheckerboardConfig(;
                common...,
                name = "metro_full_bipolar_pattern_beta05_zero_bias",
                temp = FT(0.015),
                temp_is_factor = true,
                inter_weight_scale = FT(0.25),
                input_internal_scale = FT(0.1),
                hidden_internal_scale = FT(0.1),
                output_internal_scale = FT(0.1),
                weight_seed = 13,
                internal_seed = 14,
                bias_seed = 22,
                base_seed = 93016,
                state_mode = :discrete,
                output_clamp_mode = :pattern,
                β = FT(0.5),
                lr = FT(0.0015),
            ),
            input_kick = true,
            freeze_inactive_input = true,
            zero_bias = true,
            notes = "same direct output-pattern clamping, but with biases removed as a possible XOR shortcut",
        ),
        StabilizedSearchConfig(
            config = LocalCheckerboardConfig(;
                common...,
                name = "metro_full_bipolar_pattern_hot",
                temp = FT(0.03),
                temp_is_factor = true,
                inter_weight_scale = FT(0.25),
                input_internal_scale = FT(0.1),
                hidden_internal_scale = FT(0.1),
                output_internal_scale = FT(0.1),
                weight_seed = 13,
                internal_seed = 14,
                bias_seed = 22,
                base_seed = 93016,
                state_mode = :discrete,
                output_clamp_mode = :pattern,
                β = FT(0.5),
                lr = FT(0.0015),
            ),
            input_kick = true,
            freeze_inactive_input = true,
            notes = "hotter full-bipolar pattern-clamped run to avoid freezing in the wrong basin",
        ),
        StabilizedSearchConfig(
            config = LocalCheckerboardConfig(;
                common...,
                name = "metro_4hidden_full_bipolar_input",
                hidden_side = 4,
                temp = FT(0.015),
                temp_is_factor = true,
                inter_weight_scale = FT(0.20),
                input_internal_scale = FT(0.1),
                hidden_internal_scale = FT(0.08),
                output_internal_scale = FT(0.1),
                weight_seed = 13,
                internal_seed = 14,
                bias_seed = 22,
                base_seed = 93016,
                state_mode = :discrete,
            ),
            input_kick = true,
            freeze_inactive_input = true,
            notes = "full bipolar input plus 4x4 hidden layer to separate all-minus from all-plus",
        ),
        StabilizedSearchConfig(
            config = LocalCheckerboardConfig(;
                common...,
                name = "metro_full_bipolar_weighted_00",
                temp = FT(0.015),
                temp_is_factor = true,
                inter_weight_scale = FT(0.25),
                input_internal_scale = FT(0.1),
                hidden_internal_scale = FT(0.1),
                output_internal_scale = FT(0.1),
                weight_seed = 13,
                internal_seed = 14,
                bias_seed = 22,
                base_seed = 93016,
                state_mode = :discrete,
            ),
            input_kick = true,
            freeze_inactive_input = true,
            case_repeats = (4, 1, 1, 1),
            notes = "full bipolar input with extra gradient weight on the hard (0,0) false case",
        ),
        StabilizedSearchConfig(
            config = LocalCheckerboardConfig(;
                common...,
                name = "metro_4hidden_full_bipolar_widefanout",
                hidden_side = 4,
                inter_radius = FT(10),
                temp = FT(0.015),
                temp_is_factor = true,
                inter_weight_scale = FT(0.08),
                input_internal_scale = FT(0.1),
                hidden_internal_scale = FT(0.05),
                output_internal_scale = FT(0.1),
                weight_seed = 13,
                internal_seed = 14,
                bias_seed = 22,
                base_seed = 93016,
                state_mode = :discrete,
            ),
            input_kick = true,
            freeze_inactive_input = true,
            notes = "4x4 hidden with wide input-hidden-output fanout; tests whether strict locality is the bottleneck",
        ),
        StabilizedSearchConfig(
            config = LocalCheckerboardConfig(;
                common...,
                name = "metro_full_bipolar_flipgrad",
                temp = FT(0.015),
                temp_is_factor = true,
                inter_weight_scale = FT(0.25),
                input_internal_scale = FT(0.1),
                hidden_internal_scale = FT(0.1),
                output_internal_scale = FT(0.1),
                weight_seed = 13,
                internal_seed = 14,
                bias_seed = 22,
                base_seed = 93016,
                state_mode = :discrete,
            ),
            input_kick = true,
            freeze_inactive_input = true,
            gradient_multiplier = -one(FT),
            notes = "full bipolar input plus opposite-gradient diagnostic",
        ),
        StabilizedSearchConfig(
            config = LocalCheckerboardConfig(;
                common...,
                name = "metro_2x2_4hidden_oldrecipe_random",
                hidden_side = 4,
                temp = FT(0.015),
                temp_is_factor = true,
                inter_weight_scale = FT(0.20),
                input_internal_scale = FT(0.1),
                hidden_internal_scale = FT(0.08),
                output_internal_scale = FT(0.1),
                weight_seed = 13,
                internal_seed = 14,
                bias_seed = 22,
                base_seed = 93016,
                state_mode = :discrete,
            ),
            notes = "same 2x2 physical XOR code, but 4x4 hidden layer for extra local capacity",
        ),
        StabilizedSearchConfig(
            config = LocalCheckerboardConfig(;
                name = "metro_T0015_J012_baseline",
                temp = FT(0.015),
                inter_weight_scale = FT(0.12),
                state_mode = :discrete,
                common...,
            ),
            notes = "reference Metropolis recipe, no projection",
        ),
        StabilizedSearchConfig(
            config = LocalCheckerboardConfig(;
                name = "metro_T0012_J012_clip",
                temp = FT(0.012),
                inter_weight_scale = FT(0.12),
                state_mode = :discrete,
                common...,
            ),
            max_abs_weight = FT(1.25),
            max_abs_bias = FT(0.5),
            notes = "weight/bias clipping to avoid runaway fields",
        ),
        StabilizedSearchConfig(
            config = LocalCheckerboardConfig(;
                name = "metro_T0010_J010_zero_bias",
                temp = FT(0.010),
                inter_weight_scale = FT(0.10),
                state_mode = :discrete,
                common...,
            ),
            max_abs_weight = FT(1.0),
            zero_bias = true,
            notes = "zero bias probe for XOR symmetry",
        ),
        StabilizedSearchConfig(
            config = LocalCheckerboardConfig(;
                name = "metro_T0015_J012_fieldnorm",
                temp = FT(0.015),
                inter_weight_scale = FT(0.12),
                state_mode = :discrete,
                common...,
            ),
            target_max_local_energy = FT(1.0),
            max_abs_bias = FT(0.5),
            notes = "normalize max local interaction energy to 1.0 after updates",
        ),
    ]
    wanted = split(get(ENV, "ISING_STABLE_XOR_CONFIGS", ""), ",")
    wanted = filter(!isempty, strip.(wanted))
    isempty(wanted) && return searches
    selected = [search for search in searches if search.config.name in wanted]
    isempty(selected) && error("No stabilized search configs matched ISING_STABLE_XOR_CONFIGS=$(join(wanted, ","))")
    return selected
end

function main()
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    outdir = get(ENV, "ISING_STABLE_XOR_DIR", joinpath(DEFAULT_RUN_ROOT, "stabilized_search_$timestamp"))
    searches = stabilized_searches()
    all_rows = Dict{String,Any}[]
    results = []
    for (idx, search) in enumerate(searches)
        println("Running stabilized search $idx/$(length(searches)): $(search.config.name)")
        result = run_stabilized_config(search, joinpath(outdir, search.config.name))
        append!(all_rows, result.rows)
        push!(results, result)
        println("best $(search.config.name): mse=$(round(result.best_mse, digits=6)) acc=$(round(result.best_acc, digits=3))")
    end
    csv_path = write_csv(joinpath(outdir, "local_checkerboard_stabilized_metrics.csv"), all_rows)
    png_path = plot_rows(joinpath(outdir, "local_checkerboard_stabilized_progress.png"), all_rows)
    md_path = write_stabilized_readme(joinpath(outdir, "README.md"), searches, results, csv_path, png_path)
    println("Saved metrics: $csv_path")
    println("Saved plot: $png_path")
    println("Saved docs: $md_path")
    return (; outdir, searches, results, csv_path, png_path, md_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
