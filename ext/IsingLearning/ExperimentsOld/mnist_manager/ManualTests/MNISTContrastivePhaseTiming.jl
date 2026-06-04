using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using Dates
using IsingLearning
using IsingLearning.InteractiveIsing
using IsingLearning.InteractiveIsing.StatefulAlgorithms
using Random
using SparseArrays
using Statistics

const WORKERS = parse.(Int, split(get(ENV, "ISING_MNIST_PHASE_WORKERS", "1,16,32"), ","))
const HIDDEN = parse(Int, get(ENV, "ISING_MNIST_PHASE_HIDDEN", "784"))
const OUTPUT_REPLICAS = parse(Int, get(ENV, "ISING_MNIST_PHASE_OUTPUT_REPLICAS", "4"))
const BATCHSIZE = parse(Int, get(ENV, "ISING_MNIST_PHASE_BATCHSIZE", "64"))
const SWEEPS = parse(Float64, get(ENV, "ISING_MNIST_PHASE_SWEEPS", "250.0"))
const TEMP = parse(Float32, get(ENV, "ISING_MNIST_PHASE_TEMP", "0.001"))
const STEPSIZE = parse(Float32, get(ENV, "ISING_MNIST_PHASE_STEPSIZE", "0.5"))
const BETA = parse(Float32, get(ENV, "ISING_MNIST_PHASE_BETA", "0.1"))
const WEIGHT_SCALE = parse(Float32, get(ENV, "ISING_MNIST_PHASE_WEIGHT_SCALE", "0.005"))
const ORDER = Symbol(get(ENV, "ISING_MNIST_PHASE_ORDER", "random"))
const OUTDIR = get(ENV, "ISING_MNIST_PHASE_DIR", joinpath(@__DIR__, "..", "runs", Dates.format(now(), "yyyymmdd_HHMMSS_phase_timing")))

mkpath(OUTDIR)

"""
    append_csv_row!(path, row)

Append one named-tuple row to a CSV file, creating the header if needed.
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

Return the hidden plus output units relaxed by MNIST training.
"""
function active_units(graph::G) where {G}
    return length(InteractiveIsing.layerrange(graph[2])) + length(InteractiveIsing.layerrange(graph[end]))
end

"""
    empty_phase_totals()

Create the mutable phase accumulator used by the timing probe.
"""
function empty_phase_totals()
    return Dict(
        :reset_input => 0.0,
        :free_relax => 0.0,
        :capture_equilibrium => 0.0,
        :plus_setup => 0.0,
        :plus_relax => 0.0,
        :capture_plus => 0.0,
        :minus_setup => 0.0,
        :minus_relax => 0.0,
        :capture_minus => 0.0,
        :gradient => 0.0,
    )
end

"""
    add_elapsed!(totals, name, f)

Run `f` and add its elapsed wall time to `totals[name]`.
"""
function add_elapsed!(totals::T, name::Symbol, f::F) where {T<:AbstractDict,F}
    start = time_ns()
    value = f()
    totals[name] += (time_ns() - start) / 1.0e9
    return value
end

function add_elapsed!(f::F, totals::T, name::Symbol) where {F,T<:AbstractDict}
    return add_elapsed!(totals, name, f)
end

"""
    timed_contrastive_step!(step, context, x, y)

Run one MNIST contrastive sample while timing each major phase.
"""
function timed_contrastive_step!(step::S, context::C, x::X, y::Y) where {S<:IsingLearning.MNISTContrastiveStep,C,X<:AbstractVector,Y<:AbstractVector}
    model = context.model
    β = step.β
    context.x .= x
    context.y .= y
    totals = empty_phase_totals()

    add_elapsed!(totals, :reset_input) do
        resetstate!(model)
        IsingLearning.apply_input(model, context.x)
    end

    add_elapsed!(totals, :free_relax) do
        IsingLearning._relax_mnist_context!(step.dynamics_algorithm, context.free_context, step.free_relaxation_steps)
    end
    add_elapsed!(totals, :capture_equilibrium) do
        context.equilibrium_state .= state(model)
    end

    add_elapsed!(totals, :plus_setup) do
        state(model) .= context.equilibrium_state
        IsingLearning.apply_input(model, context.x)
        IsingLearning.apply_targets(model, context.y)
        IsingLearning.set_clamping_beta!(model, β)
    end
    add_elapsed!(totals, :plus_relax) do
        IsingLearning._relax_mnist_context!(step.nudged_dynamics_algorithm, context.nudged_context, step.nudged_relaxation_steps)
    end
    add_elapsed!(totals, :capture_plus) do
        context.plus_state .= state(model)
    end

    add_elapsed!(totals, :minus_setup) do
        state(model) .= context.equilibrium_state
        IsingLearning.apply_input(model, context.x)
        IsingLearning.apply_targets(model, context.y)
        IsingLearning.set_clamping_beta!(model, -β)
    end
    add_elapsed!(totals, :minus_relax) do
        IsingLearning._relax_mnist_context!(step.nudged_dynamics_algorithm, context.nudged_context, step.nudged_relaxation_steps)
    end
    add_elapsed!(totals, :capture_minus) do
        context.minus_state .= state(model)
    end

    add_elapsed!(totals, :gradient) do
        IsingLearning.set_clamping_beta!(model, zero(β))
        IsingLearning.contrastive_gradient(model, context.plus_state, context.minus_state, β; buffers = context.buffers)
    end

    return totals
end

"""
    merge_totals!(dest, src)

Accumulate phase timings from `src` into `dest`.
"""
function merge_totals!(dest::D, src::S) where {D<:AbstractDict,S<:AbstractDict}
    for (key, value) in src
        dest[key] += value
    end
    return dest
end

"""
    build_layer()

Construct the MNIST graph/layer pair used by the phase timing probe.
"""
function build_layer()
    graph = MNISTArchitecture(
        hidden = HIDDEN,
        output_replicas = OUTPUT_REPLICAS,
        precision = Float32,
        weight_scale = WEIGHT_SCALE,
        rng = Random.MersenneTwister(110_000),
    )
    temp!(graph, TEMP)
    relaxation = max(1, round(Int, SWEEPS * active_units(graph)))
    dynamics = LocalLangevin(stepsize = STEPSIZE, max_drift_fraction = 0.15f0, adjusted = false, order = ORDER)
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
    build_contexts(layer, ncontexts)

Create independent model contexts for the phase timing probe.
"""
function build_contexts(layer::L, ncontexts::I) where {L<:LayeredIsingGraphLayer,I<:Integer}
    step = IsingLearning.MNISTContrastiveStep(layer)
    contexts = Vector{Any}(undef, Int(ncontexts))
    for idx in 1:Int(ncontexts)
        graph = deepcopy(layer.model_graph)
        temp!(graph, TEMP)
        contexts[idx] = StatefulAlgorithms.init(step, (; model = graph))
    end
    return step, contexts
end

"""
    job_range(worker_idx, nworkers, njobs)

Return the contiguous job range assigned to one worker in the manual static run.
"""
function job_range(worker_idx::I, nworkers::N, njobs::J) where {I<:Integer,N<:Integer,J<:Integer}
    first_idx = ((Int(worker_idx) - 1) * Int(njobs)) ÷ Int(nworkers) + 1
    last_idx = (Int(worker_idx) * Int(njobs)) ÷ Int(nworkers)
    return first_idx:last_idx
end

"""
    run_config(nworkers)

Run one static manual batch and report time by contrastive phase.
"""
function run_config(nworkers::I) where {I<:Integer}
    setup = build_layer()
    x, y = load_mnist_arrays(setup.layer; split = :train, limit = BATCHSIZE)
    step, contexts = build_contexts(setup.layer, nworkers)

    # Warm the hot generated paths outside the timed section.
    timed_contrastive_step!(step, contexts[1], view(x, :, 1), view(y, :, 1))
    IsingLearning.zero_buffer!(contexts[1].buffers)

    worker_totals = [empty_phase_totals() for _ in 1:Int(nworkers)]
    total_seconds = @elapsed begin
        Threads.@threads :static for worker_idx in 1:Int(nworkers)
            context = contexts[worker_idx]
            totals = worker_totals[worker_idx]
            for sample_idx in job_range(worker_idx, nworkers, BATCHSIZE)
                merge_totals!(totals, timed_contrastive_step!(step, context, view(x, :, sample_idx), view(y, :, sample_idx)))
            end
        end
    end

    summed = empty_phase_totals()
    for totals in worker_totals
        merge_totals!(summed, totals)
    end

    max_worker_phase = Dict(key => maximum(totals[key] for totals in worker_totals) for key in keys(summed))
    total_phase = sum(values(summed))
    relax_sum = summed[:free_relax] + summed[:plus_relax] + summed[:minus_relax]
    setup_sum = summed[:reset_input] + summed[:plus_setup] + summed[:minus_setup]
    capture_sum = summed[:capture_equilibrium] + summed[:capture_plus] + summed[:capture_minus]
    gradient_sum = summed[:gradient]

    row = (;
        timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        workers = Int(nworkers),
        threads = Threads.nthreads(),
        hidden = HIDDEN,
        batchsize = BATCHSIZE,
        sweeps = SWEEPS,
        order = ORDER,
        relaxation = setup.relaxation,
        graph_states = InteractiveIsing.nstates(setup.graph),
        graph_edges = length(SparseArrays.getnzval(InteractiveIsing.adj(setup.graph))),
        total_seconds,
        summed_phase_seconds = total_phase,
        relax_sum,
        setup_sum,
        capture_sum,
        gradient_sum,
        max_worker_relax = max_worker_phase[:free_relax] + max_worker_phase[:plus_relax] + max_worker_phase[:minus_relax],
        max_worker_gradient = max_worker_phase[:gradient],
        free_relax = summed[:free_relax],
        plus_relax = summed[:plus_relax],
        minus_relax = summed[:minus_relax],
        gradient = gradient_sum,
    )
    append_csv_row!(joinpath(OUTDIR, "mnist_contrastive_phase_timing.csv"), row)
    println(row)
    flush(stdout)
    return row
end

"""
    main()

Time the real contrastive MNIST worker phases without ProcessManager scheduling.
"""
function main()
    println(
        "MNIST contrastive phase timing workers=", WORKERS,
        " threads=", Threads.nthreads(),
        " hidden=", HIDDEN,
        " batchsize=", BATCHSIZE,
        " sweeps=", SWEEPS,
        " order=", ORDER,
    )
    for nworkers in WORKERS
        run_config(nworkers)
    end
    println("Saved outputs in ", OUTDIR)
end

main()
