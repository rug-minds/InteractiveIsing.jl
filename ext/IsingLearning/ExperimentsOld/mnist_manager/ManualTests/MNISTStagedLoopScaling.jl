using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using Dates
using IsingLearning
using IsingLearning.InteractiveIsing
using IsingLearning.InteractiveIsing.Processes
using Random
using SparseArrays
using Statistics

const WORKERS = parse.(Int, split(get(ENV, "ISING_MNIST_STAGED_WORKERS", "1,16,32"), ","))
const HIDDEN = parse(Int, get(ENV, "ISING_MNIST_STAGED_HIDDEN", "784"))
const OUTPUT_REPLICAS = parse(Int, get(ENV, "ISING_MNIST_STAGED_OUTPUT_REPLICAS", "4"))
const BATCHSIZE = parse(Int, get(ENV, "ISING_MNIST_STAGED_BATCHSIZE", "64"))
const SWEEPS = parse(Float64, get(ENV, "ISING_MNIST_STAGED_SWEEPS", "100.0"))
const TEMP = parse(Float32, get(ENV, "ISING_MNIST_STAGED_TEMP", "0.001"))
const STEPSIZE = parse(Float32, get(ENV, "ISING_MNIST_STAGED_STEPSIZE", "0.5"))
const BETA = parse(Float32, get(ENV, "ISING_MNIST_STAGED_BETA", "0.1"))
const WEIGHT_SCALE = parse(Float32, get(ENV, "ISING_MNIST_STAGED_WEIGHT_SCALE", "0.005"))
const ORDER = Symbol(get(ENV, "ISING_MNIST_STAGED_ORDER", "cyclic"))
const ADJ_MODE = Symbol(get(ENV, "ISING_MNIST_STAGED_ADJ_MODE", "copied"))
const BIAS_MODE = Symbol(get(ENV, "ISING_MNIST_STAGED_BIAS_MODE", "copied"))
const INPUT_MODE = Symbol(get(ENV, "ISING_MNIST_STAGED_INPUT_MODE", "state"))
const STAGES = Tuple(Symbol.(split(get(ENV, "ISING_MNIST_STAGED_STAGES", "manual_field_update,manual_prepost_update,manual_prepost_noise,manual_prepost_noise_refresh,one_relax,reset_input_one_relax,three_relax,three_relax_capture,three_relax_clamp,full"), ",")))
const OUTDIR = get(ENV, "ISING_MNIST_STAGED_DIR", joinpath(@__DIR__, "..", "runs", Dates.format(now(), "yyyymmdd_HHMMSS_staged_loop_scaling")))

mkpath(OUTDIR)

"""
    StagedMNISTStep{Stage}

MNIST worker step with progressively more of the real contrastive-learning
body enabled. `Stage` is part of the type so each stage compiles as its own
specialized `ProcessAlgorithm`.
"""
struct StagedMNISTStep{Stage,D,N,T} <: Processes.ProcessAlgorithm
    dynamics_algorithm::D
    nudged_dynamics_algorithm::N
    β::T
    input_dim::Int
    output_dim::Int
    free_relaxation_steps::Int
    nudged_relaxation_steps::Int
end

"""
    StagedMNISTStep(stage, layer)

Construct a staged MNIST algorithm from the same layer data used by the normal
MNIST contrastive worker.
"""
function StagedMNISTStep(stage::S, layer::L) where {S<:Symbol,L<:LayeredIsingGraphLayer}
    return StagedMNISTStep{stage,typeof(layer.dynamics_algorithm),typeof(layer.nudged_dynamics_algorithm),typeof(layer.β)}(
        deepcopy(layer.dynamics_algorithm),
        deepcopy(layer.nudged_dynamics_algorithm),
        layer.β,
        length(layer.input_layer),
        length(layer.output_layer),
        layer.free_relaxation_steps,
        layer.nudged_relaxation_steps,
    )
end

"""
    append_csv_row!(path, row)

Append a named-tuple row to a CSV file, creating the header when the file is new.
"""
function append_csv_row!(path::P, row::R) where {P<:AbstractString,R<:NamedTuple}
    names = propertynames(row)
    needs_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        needs_header && println(io, join(names, ","))
        println(io, join((getproperty(row, name) for name in names), ","))
    end
    return path
end

"""
    active_units(graph)

Return the number of hidden and output units updated during MNIST relaxation.
"""
function active_units(graph::G) where {G}
    return length(InteractiveIsing.layerrange(graph[2])) + length(InteractiveIsing.layerrange(graph[end]))
end

"""
    job_range(worker_idx, nworkers, njobs)

Return the contiguous sample range assigned to one static worker.
"""
function job_range(worker_idx::I, nworkers::N, njobs::J) where {I<:Integer,N<:Integer,J<:Integer}
    first_idx = ((Int(worker_idx) - 1) * Int(njobs)) ÷ Int(nworkers) + 1
    last_idx = (Int(worker_idx) * Int(njobs)) ÷ Int(nworkers)
    return first_idx:last_idx
end

"""
    build_layer()

Construct the MNIST graph and layer used for all staged runs.
"""
function build_layer()
    input_b = INPUT_MODE === :field ? (g -> zeros(Float32, InteractiveIsing.nstates(g))) :
        INPUT_MODE === :state ? nothing :
        throw(ArgumentError("ISING_MNIST_STAGED_INPUT_MODE must be state or field, got $(repr(INPUT_MODE))."))
    graph = MNISTArchitecture(
        hidden = HIDDEN,
        output_replicas = OUTPUT_REPLICAS,
        precision = Float32,
        weight_scale = WEIGHT_SCALE,
        rng = Random.MersenneTwister(140_000),
        input_b = input_b,
    )
    temp!(graph, TEMP)
    relaxation = max(1, round(Int, SWEEPS * active_units(graph)))
    dynamics = LocalLangevin(
        stepsize = STEPSIZE,
        max_drift_fraction = 0.15f0,
        adjusted = false,
        order = ORDER,
    )
    layer = MNISTLayer(
        graph = graph,
        β = BETA,
        relaxation_steps = relaxation,
        dynamics_algorithm = dynamics,
        nudged_dynamics_algorithm = deepcopy(dynamics),
        validation_algorithm = deepcopy(dynamics),
    )
    return (; graph, layer, relaxation)
end

"""
    Processes.init(step, context)

Create the reusable context for a staged MNIST worker. Every stage gets the same
context layout so timing differences come from `step!`, not context shape.
"""
function Processes.init(step::S, context) where {Stage,D,N,T,S<:StagedMNISTStep{Stage,D,N,T}}
    model = context.model
    GType = eltype(model)
    x = get(context, :x, zeros(GType, step.input_dim))
    y = get(context, :y, zeros(GType, step.output_dim))
    buffers = get(context, :buffers, IsingLearning.gradient_buffer(model))
    equilibrium_state = get(context, :equilibrium_state, copy(state(model)))
    plus_state = get(context, :plus_state, similar(equilibrium_state))
    minus_state = get(context, :minus_state, similar(equilibrium_state))
    input_pattern = get(context, :input_pattern, isnothing(IsingLearning._mnist_input_magfield(model)) ? nothing : zeros(GType, InteractiveIsing.nstates(model)))
    free_context = Processes.init(step.dynamics_algorithm, (; model))
    nudged_context = Processes.init(step.nudged_dynamics_algorithm, (; model))
    return (; model, x, y, buffers, equilibrium_state, plus_state, minus_state, input_pattern, free_context, nudged_context)
end

"""
    manual_relaxation_ladder!(step, context, nsteps)

Run a hand-written local update ladder on the same graph and Hamiltonian as
`LocalLangevin`. These stages isolate which pieces inside relaxation hurt
parallel scaling before the full `LocalLangevin` step is used.
"""
function manual_relaxation_ladder!(step::S, context::C, nsteps::I) where {Stage,D,N,T,S<:StagedMNISTStep{Stage,D,N,T},C,I<:Integer}
    model = context.model
    local_context = context.free_context
    hamiltonian = local_context.hamiltonian
    active_spins = local_context.active_spins
    dH_prealloc = local_context.dH_prealloc
    rng = local_context.rng
    spins = state(model)
    dh = InteractiveIsing.d_iH()
    nactive = length(active_spins)
    nactive == 0 && return context

    step_size = eltype(model)(STEPSIZE)
    noise_scale = sqrt(eltype(model)(2) * step_size * max(eltype(model)(TEMP), zero(eltype(model))))
    low_state = -one(eltype(model))
    high_state = one(eltype(model))

    @inbounds for step_idx in 1:Int(nsteps)
        if Stage === :manual_prepost_noise_refresh && (step_idx == 1 || mod1(step_idx, nactive) == 1)
            for spin_idx in active_spins
                derivative = InteractiveIsing.calculate(dh, hamiltonian, model, spin_idx)
                dH_prealloc[spin_idx] = eltype(model)(derivative)
            end
        end

        spin_idx = active_spins[mod1(step_idx, nactive)]
        derivative = eltype(model)(InteractiveIsing.calculate(dh, hamiltonian, model, spin_idx))
        old_state = spins[spin_idx]
        noise = (Stage === :manual_prepost_noise || Stage === :manual_prepost_noise_refresh) ? noise_scale * randn(rng, eltype(model)) : zero(eltype(model))
        spins[spin_idx] = clamp(old_state - step_size * derivative + noise, low_state, high_state)

        if Stage === :manual_prepost_update || Stage === :manual_prepost_noise || Stage === :manual_prepost_noise_refresh
            post_derivative = InteractiveIsing.calculate(dh, hamiltonian, model, spin_idx)
            dH_prealloc[spin_idx] = eltype(model)(post_derivative)
        end
    end
    return context
end

"""
    Processes.step!(step, context)

Run one staged MNIST sample. Stages are cumulative:
manual stages add pieces inside relaxation, `one_relax` switches to the full
`LocalLangevin` relaxation, `reset_input_one_relax` adds input setup,
`three_relax` adds free/plus/minus work, `three_relax_capture` adds state
copies, `three_relax_clamp` adds target clamping, and `full` adds gradient
accumulation.
"""
function Processes.step!(step::S, context::C) where {Stage,D,N,T,S<:StagedMNISTStep{Stage,D,N,T},C}
    model = context.model
    β = step.β

    if Stage === :manual_field_update || Stage === :manual_prepost_update || Stage === :manual_prepost_noise || Stage === :manual_prepost_noise_refresh
        manual_relaxation_ladder!(step, context, step.free_relaxation_steps)
        return nothing
    end

    if Stage === :one_relax
        IsingLearning._relax_mnist_context!(step.dynamics_algorithm, context.free_context, step.free_relaxation_steps)
        return nothing
    end

    resetstate!(model)
    IsingLearning._apply_mnist_context_input!(model, context)
    IsingLearning._relax_mnist_context!(step.dynamics_algorithm, context.free_context, step.free_relaxation_steps)

    Stage === :reset_input_one_relax && return nothing

    if Stage === :three_relax
        resetstate!(model)
        IsingLearning._apply_mnist_context_input!(model, context)
        IsingLearning._relax_mnist_context!(step.nudged_dynamics_algorithm, context.nudged_context, step.nudged_relaxation_steps)
        resetstate!(model)
        IsingLearning._apply_mnist_context_input!(model, context)
        IsingLearning._relax_mnist_context!(step.nudged_dynamics_algorithm, context.nudged_context, step.nudged_relaxation_steps)
        return nothing
    end

    context.equilibrium_state .= state(model)

    state(model) .= context.equilibrium_state
    IsingLearning._apply_mnist_context_input!(model, context)
    if Stage === :three_relax_clamp || Stage === :full
        IsingLearning.apply_targets(model, context.y)
        IsingLearning.set_clamping_beta!(model, β)
    end
    IsingLearning._relax_mnist_context!(step.nudged_dynamics_algorithm, context.nudged_context, step.nudged_relaxation_steps)
    context.plus_state .= state(model)

    state(model) .= context.equilibrium_state
    IsingLearning._apply_mnist_context_input!(model, context)
    if Stage === :three_relax_clamp || Stage === :full
        IsingLearning.apply_targets(model, context.y)
        IsingLearning.set_clamping_beta!(model, -β)
    end
    IsingLearning._relax_mnist_context!(step.nudged_dynamics_algorithm, context.nudged_context, step.nudged_relaxation_steps)
    context.minus_state .= state(model)

    if Stage === :full
        IsingLearning.set_clamping_beta!(model, zero(β))
        IsingLearning.contrastive_gradient(model, context.plus_state, context.minus_state, β; buffers = context.buffers)
    end
    return nothing
end

"""
    build_contexts(layer, step, ncontexts)

Create independent graph contexts for one staged algorithm.
"""
function build_contexts(layer::L, step::S, ncontexts::I) where {L<:LayeredIsingGraphLayer,S<:StagedMNISTStep,I<:Integer}
    contexts = Vector{Any}(undef, Int(ncontexts))
    fixed_input = zeros(eltype(layer.model_graph), step.input_dim)
    shared_adj = adj(layer.model_graph)
    shared_bias = IsingLearning._mnist_base_magfield(layer.model_graph).b
    has_input_field = !isnothing(IsingLearning._mnist_input_magfield(layer.model_graph))
    for idx in 1:Int(ncontexts)
        graph = if ADJ_MODE === :shared || BIAS_MODE === :shared
            base_bias = BIAS_MODE === :shared ? shared_bias :
                BIAS_MODE === :copied ? copy(shared_bias) :
                throw(ArgumentError("ISING_MNIST_STAGED_BIAS_MODE must be copied or shared, got $(repr(BIAS_MODE))."))
            input_bias = has_input_field ? zeros(eltype(layer.model_graph), InteractiveIsing.nstates(layer.model_graph)) : nothing
            MNISTArchitecture(
                hidden = HIDDEN,
                output_replicas = OUTPUT_REPLICAS,
                precision = eltype(layer.model_graph),
                weight_scale = WEIGHT_SCALE,
                rng = Random.MersenneTwister(150_000 + idx),
                adj = ADJ_MODE === :shared ? shared_adj : nothing,
                b = base_bias,
                input_b = input_bias,
            )
        else
            deepcopy(layer.model_graph)
        end
        temp!(graph, TEMP)
        if ADJ_MODE !== :copied && ADJ_MODE !== :shared
            throw(ArgumentError("ISING_MNIST_STAGED_ADJ_MODE must be copied or shared, got $(repr(ADJ_MODE))."))
        end
        IsingLearning.apply_input(graph, fixed_input)
        contexts[idx] = Processes.init(step, (; model = graph))
    end
    return contexts
end

"""
    run_stage_batch!(step, contexts, x, y)

Run a static staged batch and return total wall time plus per-worker timings.
"""
function run_stage_batch!(step::S, contexts::C, x::X, y::Y) where {S<:StagedMNISTStep,C<:AbstractVector,X<:AbstractMatrix,Y<:AbstractMatrix}
    task_seconds = zeros(Float64, length(contexts))
    total_seconds = @elapsed begin
        Threads.@threads :static for worker_idx in eachindex(contexts)
            context = contexts[worker_idx]
            worker_start = time_ns()
            for sample_idx in job_range(worker_idx, length(contexts), size(x, 2))
                context.x .= view(x, :, sample_idx)
                context.y .= view(y, :, sample_idx)
                if !isnothing(context.input_pattern)
                    IsingLearning.precompute_mnist_input_pattern!(context.model, context.input_pattern, view(x, :, sample_idx))
                end
                Processes.step!(step, context)
            end
            task_seconds[worker_idx] = (time_ns() - worker_start) / 1.0e9
        end
    end
    return (; total_seconds, task_seconds)
end

"""
    run_config(stage, nworkers)

Measure one stage and one worker count.
"""
function run_config(stage::S, nworkers::I) where {S<:Symbol,I<:Integer}
    setup = build_layer()
    x, y = load_mnist_arrays(setup.layer; split = :train, limit = BATCHSIZE)
    step = StagedMNISTStep(stage, setup.layer)
    contexts = build_contexts(setup.layer, step, nworkers)

    context = contexts[1]
    context.x .= view(x, :, 1)
    context.y .= view(y, :, 1)
    if !isnothing(context.input_pattern)
        IsingLearning.precompute_mnist_input_pattern!(context.model, context.input_pattern, view(x, :, 1))
    end
    Processes.step!(step, context)
    IsingLearning.zero_buffer!(context.buffers)

    timed = run_stage_batch!(step, contexts, x, y)
    row = (;
        timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        stage,
        workers = Int(nworkers),
        threads = Threads.nthreads(),
        hidden = HIDDEN,
        output_replicas = OUTPUT_REPLICAS,
        batchsize = BATCHSIZE,
        sweeps = SWEEPS,
        order = ORDER,
        adj_mode = ADJ_MODE,
        bias_mode = BIAS_MODE,
        input_mode = INPUT_MODE,
        relaxation_steps = setup.relaxation,
        graph_states = InteractiveIsing.nstates(setup.graph),
        graph_edges = length(SparseArrays.getnzval(InteractiveIsing.adj(setup.graph))),
        total_seconds = timed.total_seconds,
        min_task_seconds = minimum(timed.task_seconds),
        mean_task_seconds = mean(timed.task_seconds),
        max_task_seconds = maximum(timed.task_seconds),
    )
    append_csv_row!(joinpath(OUTDIR, "mnist_staged_loop_scaling.csv"), row)
    println(row)
    flush(stdout)
    return row
end

"""
    main()

Run staged MNIST worker scaling benchmarks.
"""
function main()
    println(
        "MNIST staged loop scaling stages=", STAGES,
        " workers=", WORKERS,
        " threads=", Threads.nthreads(),
        " hidden=", HIDDEN,
        " batchsize=", BATCHSIZE,
        " sweeps=", SWEEPS,
        " order=", ORDER,
        " adj_mode=", ADJ_MODE,
        " bias_mode=", BIAS_MODE,
        " input_mode=", INPUT_MODE,
    )
    for stage in STAGES
        for nworkers in WORKERS
            run_config(stage, nworkers)
        end
    end
    println("Saved outputs in ", OUTDIR)
end

main()
