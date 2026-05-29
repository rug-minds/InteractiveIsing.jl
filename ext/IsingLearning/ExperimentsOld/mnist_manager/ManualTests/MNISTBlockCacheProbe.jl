using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using Dates
using IsingLearning
using IsingLearning.InteractiveIsing
using LinearAlgebra
using LoopVectorization
using Random
using SparseArrays
using Statistics

const WORKERS = parse.(Int, split(get(ENV, "ISING_MNIST_BLOCK_CACHE_WORKERS", "16,32"), ","))
const HIDDENS = parse.(Int, split(get(ENV, "ISING_MNIST_BLOCK_CACHE_HIDDENS", "120,784"), ","))
const OUTPUT_REPLICAS = parse(Int, get(ENV, "ISING_MNIST_BLOCK_CACHE_OUTPUT_REPLICAS", "4"))
const SWEEPS = parse(Float64, get(ENV, "ISING_MNIST_BLOCK_CACHE_SWEEPS", "1000.0"))
const TEMP = parse(Float32, get(ENV, "ISING_MNIST_BLOCK_CACHE_TEMP", "0.001"))
const STEPSIZE = parse(Float32, get(ENV, "ISING_MNIST_BLOCK_CACHE_STEPSIZE", "0.5"))
const WEIGHT_SCALE = parse(Float32, get(ENV, "ISING_MNIST_BLOCK_CACHE_WEIGHT_SCALE", "0.005"))
const ORDER_MODE = Symbol(get(ENV, "ISING_MNIST_BLOCK_CACHE_ORDER", "random"))
const SCHEDULE_MODE = Symbol(get(ENV, "ISING_MNIST_BLOCK_CACHE_SCHEDULE", "spawn"))
const OUTDIR = get(ENV, "ISING_MNIST_BLOCK_CACHE_DIR", joinpath(@__DIR__, "..", "runs", Dates.format(now(), "yyyymmdd_HHMMSS_block_cache_probe")))

mkpath(OUTDIR)

mutable struct MNISTBlockProbeContext{W1,W2,W3,VH,VO,FH,FO,R}
    Wxh::W1
    Who::W2
    Woh::W3
    h::VH
    o::VO
    field_h::FH
    field_o::FO
    rng::R
end

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
    proposal_state(old_state, field)

Small deterministic bounded proposal for a field-update cost model.
"""
@inline function proposal_state(old_state::T, field::T) where {T<:AbstractFloat}
    return clamp(old_state - T(STEPSIZE) * field, -one(T), one(T))
end

"""
    graph_bias(graph)

Return the graph magnetic-field bias vector.
"""
function graph_bias(graph::G) where {G}
    return InteractiveIsing.getparam(graph.hamiltonian, InteractiveIsing.MagField, :b)
end

"""
    dense_blocks(graph)

Extract dense input-hidden and hidden-output weight blocks from the MNIST graph.
The blocks are oriented so hidden fields use `Wxh' * x` and output fields use
`Who' * h`.
"""
function dense_blocks(graph::G) where {G}
    input = collect(InteractiveIsing.layerrange(graph[1]))
    hidden = collect(InteractiveIsing.layerrange(graph[2]))
    output = collect(InteractiveIsing.layerrange(graph[end]))
    sp = InteractiveIsing.adj(graph).sp
    Wxh = Matrix(sp[input, hidden])
    Who = Matrix(sp[hidden, output])
    b = graph_bias(graph)
    return (;
        Wxh,
        Who,
        b_h = copy(view(b, hidden)),
        b_o = copy(view(b, output)),
        input,
        hidden,
        output,
    )
end

"""
    build_prototype(hidden)

Build one MNIST graph, extract dense blocks, and normalize one MNIST sample.
"""
function build_prototype(hidden::I) where {I<:Integer}
    graph = MNISTArchitecture(
        hidden = Int(hidden),
        output_replicas = OUTPUT_REPLICAS,
        precision = Float32,
        weight_scale = WEIGHT_SCALE,
        rng = Random.MersenneTwister(90_000 + Int(hidden)),
    )
    temp!(graph, TEMP)
    layer = MNISTLayer(graph = graph)
    x, _ = load_mnist_arrays(layer; split = :train, limit = 1)
    blocks = dense_blocks(graph)
    nactive = length(blocks.hidden) + length(blocks.output)
    nsteps = max(1, round(Int, SWEEPS * nactive))
    return (; graph, x = copy(view(x, :, 1)), blocks, nsteps)
end

"""
    init_fields!(field_h, field_o, Wxh, Who, x, h, o, b_h, b_o)

Initialize hidden/output local fields for the block-cache probe.
"""
function init_fields!(
    field_h::FH,
    field_o::FO,
    Wxh::W1,
    Who::W2,
    x::X,
    h::H,
    o::O,
    b_h::BH,
    b_o::BO,
) where {FH<:AbstractVector,FO<:AbstractVector,W1<:AbstractMatrix,W2<:AbstractMatrix,X<:AbstractVector,H<:AbstractVector,O<:AbstractVector,BH<:AbstractVector,BO<:AbstractVector}
    copyto!(field_h, b_h)
    mul!(field_h, transpose(Wxh), x, one(eltype(field_h)), one(eltype(field_h)))
    mul!(field_h, Who, o, one(eltype(field_h)), one(eltype(field_h)))
    copyto!(field_o, b_o)
    mul!(field_o, transpose(Who), h, one(eltype(field_o)), one(eltype(field_o)))
    return field_h, field_o
end

"""
    build_contexts(prototype, ncontexts; seed)

Create independent dense block-cache contexts.
"""
function build_contexts(prototype::P, ncontexts::I; seed::S = 1) where {P,I<:Integer,S<:Integer}
    blocks = prototype.blocks
    contexts = MNISTBlockProbeContext[]
    sizehint!(contexts, Int(ncontexts))
    for idx in 1:Int(ncontexts)
        Wxh = copy(blocks.Wxh)
        Who = copy(blocks.Who)
        Woh = copy(transpose(Who))
        h = zeros(Float32, size(Who, 1))
        o = zeros(Float32, size(Who, 2))
        randn!(Random.MersenneTwister(seed + 10_000 + idx), h)
        randn!(Random.MersenneTwister(seed + 20_000 + idx), o)
        h .= clamp.(0.05f0 .* h, -1f0, 1f0)
        o .= clamp.(0.05f0 .* o, -1f0, 1f0)
        field_h = similar(h)
        field_o = similar(o)
        init_fields!(field_h, field_o, Wxh, Who, prototype.x, h, o, blocks.b_h, blocks.b_o)
        push!(contexts, MNISTBlockProbeContext(Wxh, Who, Woh, h, o, field_h, field_o, Random.MersenneTwister(seed + idx)))
    end
    return contexts
end

"""
    update_hidden!(context, idx)

Apply one hidden-spin update and refresh output fields through a contiguous row
of `Who`.
"""
@inline function update_hidden!(context::C, idx::I) where {C<:MNISTBlockProbeContext,I<:Integer}
    old_state = @inbounds context.h[idx]
    field = @inbounds context.field_h[idx]
    new_state = proposal_state(old_state, field)
    delta = new_state - old_state
    @inbounds context.h[idx] = new_state
    Woh = context.Woh
    field_o = context.field_o
    @turbo for j in eachindex(field_o)
        field_o[j] += Woh[j, idx] * delta
    end
    return field
end

"""
    update_output!(context, idx)

Apply one output-spin update and refresh hidden fields through a contiguous
column traversal of `Who`.
"""
@inline function update_output!(context::C, idx::I) where {C<:MNISTBlockProbeContext,I<:Integer}
    old_state = @inbounds context.o[idx]
    field = @inbounds context.field_o[idx]
    new_state = proposal_state(old_state, field)
    delta = new_state - old_state
    @inbounds context.o[idx] = new_state
    Who = context.Who
    field_h = context.field_h
    @turbo for i in eachindex(field_h)
        field_h[i] += Who[i, idx] * delta
    end
    return field
end

"""
    run_block_probe!(context, nsteps)

Run local updates on hidden/output states using block-cached fields.
"""
function run_block_probe!(context::C, nsteps::I) where {C<:MNISTBlockProbeContext,I<:Integer}
    nh = length(context.h)
    no = length(context.o)
    ntotal = nh + no
    checksum = zero(eltype(context.h))
    if ORDER_MODE === :shuffle
        order = collect(1:ntotal)
        done = 0
        @inbounds while done < Int(nsteps)
            Random.shuffle!(context.rng, order)
            for pos in order
                done >= Int(nsteps) && break
                if pos <= nh
                    checksum += update_hidden!(context, pos)
                else
                    checksum += update_output!(context, pos - nh)
                end
                done += 1
            end
        end
        return checksum + sum(context.h) + sum(context.o)
    end
    @inbounds for step_idx in 1:Int(nsteps)
        pos = ORDER_MODE === :cyclic ? mod1(step_idx, ntotal) : rand(context.rng, 1:ntotal)
        if pos <= nh
            checksum += update_hidden!(context, pos)
        else
            checksum += update_output!(context, pos - nh)
        end
    end
    return checksum + sum(context.h) + sum(context.o)
end

"""
    timed_block_probe!(context, nsteps)

Measure one block-cache context.
"""
function timed_block_probe!(context::C, nsteps::I) where {C<:MNISTBlockProbeContext,I<:Integer}
    result = Ref{Any}(nothing)
    seconds = @elapsed result[] = run_block_probe!(context, nsteps)
    return (; seconds, result = result[])
end

"""
    run_parallel_probe!(contexts, nsteps)

Spawn one task per context and run block-cache updates.
"""
function run_parallel_probe!(contexts::C, nsteps::I) where {C<:AbstractVector,I<:Integer}
    task_seconds = zeros(Float64, length(contexts))
    outputs = Vector{Any}(undef, length(contexts))
    total_seconds = if SCHEDULE_MODE === :spawn
        @elapsed begin
        tasks = map(eachindex(contexts)) do idx
            Threads.@spawn (idx = $idx, timed = timed_block_probe!($(contexts[idx]), $nsteps))
        end
        for task in tasks
            output = fetch(task)
            task_seconds[output.idx] = output.timed.seconds
            outputs[output.idx] = output.timed.result
        end
        end
    elseif SCHEDULE_MODE === :static
        @elapsed begin
            Threads.@threads :static for idx in eachindex(contexts)
                timed = timed_block_probe!(contexts[idx], nsteps)
                task_seconds[idx] = timed.seconds
                outputs[idx] = timed.result
            end
        end
    else
        throw(ArgumentError("unknown schedule mode $SCHEDULE_MODE"))
    end
    return (; total_seconds, task_seconds, checksum = sum(outputs))
end

"""
    run_config(hidden, ncontexts)

Run one block-cache probe configuration.
"""
function run_config(hidden::H, ncontexts::N) where {H<:Integer,N<:Integer}
    prototype = build_prototype(hidden)
    single_context = only(build_contexts(prototype, 1; seed = 10_000))
    timed_block_probe!(single_context, 1)
    single = timed_block_probe!(single_context, prototype.nsteps)

    contexts = build_contexts(prototype, ncontexts; seed = 20_000)
    for context in contexts
        timed_block_probe!(context, 1)
    end
    parallel = run_parallel_probe!(contexts, prototype.nsteps)

    serial_all = single.seconds * Int(ncontexts)
    row = (;
        timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        hidden = Int(hidden),
        contexts = Int(ncontexts),
        threads = Threads.nthreads(),
        order = string(ORDER_MODE),
        schedule = string(SCHEDULE_MODE),
        sweeps = SWEEPS,
        nsteps = prototype.nsteps,
        graph_states = InteractiveIsing.nstates(prototype.graph),
        graph_edges = length(SparseArrays.getnzval(InteractiveIsing.adj(prototype.graph))),
        single_seconds = single.seconds,
        parallel_seconds = parallel.total_seconds,
        min_task_seconds = minimum(parallel.task_seconds),
        mean_task_seconds = mean(parallel.task_seconds),
        max_task_seconds = maximum(parallel.task_seconds),
        speedup_vs_serial_all = serial_all / parallel.total_seconds,
    )
    append_csv_row!(joinpath(OUTDIR, "mnist_block_cache_probe.csv"), row)
    println(row)
    flush(stdout)
    return row
end

"""
    main()

Probe dense layer-block cached MNIST hidden/output updates.
"""
function main()
    println(
        "MNIST block-cache probe hiddens=", HIDDENS,
        " workers=", WORKERS,
        " threads=", Threads.nthreads(),
        " order=", ORDER_MODE,
        " schedule=", SCHEDULE_MODE,
        " sweeps=", SWEEPS,
    )
    for hidden in HIDDENS
        for ncontexts in WORKERS
            run_config(hidden, ncontexts)
        end
    end
    println("Saved outputs in ", OUTDIR)
end

main()
