using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using Dates
using IsingLearning
using IsingLearning.InteractiveIsing
using Random
using SparseArrays
using Statistics

const WORKERS = parse.(Int, split(get(ENV, "ISING_MNIST_FIELD_CACHE_WORKERS", "16,32"), ","))
const HIDDENS = parse.(Int, split(get(ENV, "ISING_MNIST_FIELD_CACHE_HIDDENS", "120,784"), ","))
const OUTPUT_REPLICAS = parse(Int, get(ENV, "ISING_MNIST_FIELD_CACHE_OUTPUT_REPLICAS", "4"))
const SWEEPS = parse(Float64, get(ENV, "ISING_MNIST_FIELD_CACHE_SWEEPS", "500.0"))
const TEMP = parse(Float32, get(ENV, "ISING_MNIST_FIELD_CACHE_TEMP", "0.001"))
const STEPSIZE = parse(Float32, get(ENV, "ISING_MNIST_FIELD_CACHE_STEPSIZE", "0.5"))
const WEIGHT_SCALE = parse(Float32, get(ENV, "ISING_MNIST_FIELD_CACHE_WEIGHT_SCALE", "0.005"))
const ORDER_MODE = Symbol(get(ENV, "ISING_MNIST_FIELD_CACHE_ORDER", "random"))
const SCHEDULE_MODE = Symbol(get(ENV, "ISING_MNIST_FIELD_CACHE_SCHEDULE", "spawn"))
const OUTDIR = get(ENV, "ISING_MNIST_FIELD_CACHE_DIR", joinpath(@__DIR__, "..", "runs", Dates.format(now(), "yyyymmdd_HHMMSS_field_cache_probe")))

mkpath(OUTDIR)

mutable struct FieldProbeContext{G,S,B,F,A,CP,CI,CV,R}
    graph::G
    spins::S
    bias::B
    fields::F
    active::A
    out_colptr::CP
    out_idx::CI
    out_val::CV
    rng::R
end

"""
    append_csv_row!(path, row)

Append one named-tuple row to a CSV file, creating the header on first write.
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

Return the number of non-input MNIST units that remain active during relaxation.
"""
function active_units(graph::G) where {G}
    return length(InteractiveIsing.layerrange(graph[2])) + length(InteractiveIsing.layerrange(graph[end]))
end

"""
    graph_bias(graph)

Return the magnetic-field bias vector used by the MNIST graph.
"""
function graph_bias(graph::G) where {G}
    return InteractiveIsing.getparam(graph.hamiltonian, InteractiveIsing.MagField, :b)
end

"""
    scan_field(sp, spins, bias, spin_idx)

Compute one local bilinear-plus-bias field by scanning the sparse CSC column.
This mirrors the expensive part of the current local derivative path.
"""
@inline function scan_field(
    sp::S,
    spins::V,
    bias::B,
    spin_idx::I,
) where {S<:SparseMatrixCSC,V<:AbstractVector,B<:AbstractVector,I<:Integer}
    rowval = SparseArrays.getrowval(sp)
    nzval = SparseArrays.getnzval(sp)
    total = @inbounds bias[spin_idx]
    @inbounds for ptr in SparseArrays.nzrange(sp, spin_idx)
        total += nzval[ptr] * spins[rowval[ptr]]
    end
    return total
end

"""
    compute_fields!(fields, sp, spins, bias)

Fill `fields` with `bias + J' * spins` using the same column orientation as
`column_contraction`. This is the field cache a specialized MNIST dynamics loop
could keep up to date.
"""
function compute_fields!(
    fields::F,
    sp::S,
    spins::V,
    bias::B,
) where {F<:AbstractVector,S<:SparseMatrixCSC,V<:AbstractVector,B<:AbstractVector}
    copyto!(fields, bias)
    rowval = SparseArrays.getrowval(sp)
    nzval = SparseArrays.getnzval(sp)
    @inbounds for col in axes(sp, 2)
        total = fields[col]
        for ptr in SparseArrays.nzrange(sp, col)
            total += nzval[ptr] * spins[rowval[ptr]]
        end
        fields[col] = total
    end
    return fields
end

"""
    update_fields_after_spin!(fields, sp, spin_idx, delta)

Apply the rank-one field-cache update caused by changing one spin by `delta`.
For symmetric MNIST graphs, the affected cached fields are exactly the row
indices stored in column `spin_idx`.
"""
function update_fields_after_spin!(
    fields::F,
    sp::S,
    spin_idx::I,
    delta::T,
) where {F<:AbstractVector,S<:SparseMatrixCSC,I<:Integer,T<:Real}
    rowval = SparseArrays.getrowval(sp)
    nzval = SparseArrays.getnzval(sp)
    @inbounds for ptr in SparseArrays.nzrange(sp, spin_idx)
        fields[rowval[ptr]] += nzval[ptr] * delta
    end
    return fields
end

"""
    sparse_outgoing_cache(sp)

Build a compressed row-style adjacency cache from the CSC sparse matrix. For a
changed spin `i`, the range `out_colptr[i]:(out_colptr[i + 1] - 1)` lists every
field index affected by that spin and the corresponding weight. This is still a
general sparse representation; it just stores the reverse traversal explicitly.
"""
function sparse_outgoing_cache(sp::S) where {S<:SparseMatrixCSC}
    n = size(sp, 1)
    counts = zeros(Int, n)
    rowval = SparseArrays.getrowval(sp)
    nzval = SparseArrays.getnzval(sp)
    @inbounds for row in rowval
        counts[row] += 1
    end

    out_colptr = Vector{Int}(undef, n + 1)
    out_colptr[1] = 1
    @inbounds for idx in 1:n
        out_colptr[idx + 1] = out_colptr[idx] + counts[idx]
    end

    out_idx = Vector{eltype(rowval)}(undef, length(rowval))
    out_val = similar(nzval)
    next = copy(out_colptr)
    @inbounds for col in axes(sp, 2)
        for ptr in SparseArrays.nzrange(sp, col)
            row = rowval[ptr]
            dest = next[row]
            out_idx[dest] = col
            out_val[dest] = nzval[ptr]
            next[row] += 1
        end
    end
    return out_colptr, out_idx, out_val
end

"""
    update_fields_after_spin_outgoing!(fields, out_colptr, out_idx, out_val, spin_idx, delta)

Update cached fields from an explicit outgoing sparse adjacency cache. This is
topology-general and avoids searching or transposing during the local step.
"""
function update_fields_after_spin_outgoing!(
    fields::F,
    out_colptr::CP,
    out_idx::IX,
    out_val::V,
    spin_idx::I,
    delta::T,
) where {F<:AbstractVector,CP<:AbstractVector,IX<:AbstractVector,V<:AbstractVector,I<:Integer,T<:Real}
    @inbounds for ptr in out_colptr[spin_idx]:(out_colptr[spin_idx + 1] - 1)
        fields[out_idx[ptr]] += out_val[ptr] * delta
    end
    return fields
end

"""
    proposal_state(old_state, field)

Use a small deterministic bounded update. This probe is a cost model for local
field evaluation and maintenance, not a replacement for `LocalLangevin`.
"""
@inline function proposal_state(old_state::T, field::T) where {T<:AbstractFloat}
    return clamp(old_state - T(STEPSIZE) * field, -one(T), one(T))
end

"""
    run_scan_probe!(context, nsteps)

Run a two-scan-per-step sparse-field loop. LocalLangevin currently evaluates the
local derivative before and after an accepted move, so this approximates the
memory traffic of the current derivative path.
"""
function run_scan_probe!(context::C, nsteps::I) where {C<:FieldProbeContext,I<:Integer}
    sp = InteractiveIsing.adj(context.graph).sp
    spins = context.spins
    bias = context.bias
    active = context.active
    rng = context.rng
    checksum = zero(eltype(spins))

    if ORDER_MODE === :shuffle
        order = collect(eachindex(active))
        done = 0
        @inbounds while done < Int(nsteps)
            Random.shuffle!(rng, order)
            for active_pos in order
                done >= Int(nsteps) && break
                spin_idx = active[active_pos]
                field = scan_field(sp, spins, bias, spin_idx)
                old_state = spins[spin_idx]
                new_state = proposal_state(old_state, field)
                spins[spin_idx] = new_state
                checksum += scan_field(sp, spins, bias, spin_idx)
                done += 1
            end
        end
        return checksum + sum(spins)
    end

    @inbounds for step_idx in 1:Int(nsteps)
        active_pos = ORDER_MODE === :cyclic ? mod1(step_idx, length(active)) : rand(rng, eachindex(active))
        spin_idx = active[active_pos]
        field = scan_field(sp, spins, bias, spin_idx)
        old_state = spins[spin_idx]
        new_state = proposal_state(old_state, field)
        spins[spin_idx] = new_state
        checksum += scan_field(sp, spins, bias, spin_idx)
    end
    return checksum + sum(spins)
end

"""
    run_cache_probe!(context, nsteps)

Run a cached-field loop. Each step reads one cached derivative and updates all
neighbor fields once after the spin changes.
"""
function run_cache_probe!(context::C, nsteps::I) where {C<:FieldProbeContext,I<:Integer}
    sp = InteractiveIsing.adj(context.graph).sp
    spins = context.spins
    fields = context.fields
    active = context.active
    rng = context.rng
    checksum = zero(eltype(spins))

    if ORDER_MODE === :shuffle
        order = collect(eachindex(active))
        done = 0
        @inbounds while done < Int(nsteps)
            Random.shuffle!(rng, order)
            for active_pos in order
                done >= Int(nsteps) && break
                spin_idx = active[active_pos]
                field = fields[spin_idx]
                old_state = spins[spin_idx]
                new_state = proposal_state(old_state, field)
                delta = new_state - old_state
                spins[spin_idx] = new_state
                update_fields_after_spin!(fields, sp, spin_idx, delta)
                checksum += fields[spin_idx]
                done += 1
            end
        end
        return checksum + sum(spins)
    end

    @inbounds for step_idx in 1:Int(nsteps)
        active_pos = ORDER_MODE === :cyclic ? mod1(step_idx, length(active)) : rand(rng, eachindex(active))
        spin_idx = active[active_pos]
        field = fields[spin_idx]
        old_state = spins[spin_idx]
        new_state = proposal_state(old_state, field)
        delta = new_state - old_state
        spins[spin_idx] = new_state
        update_fields_after_spin!(fields, sp, spin_idx, delta)
        checksum += fields[spin_idx]
    end
    return checksum + sum(spins)
end

"""
    run_outgoing_cache_probe!(context, nsteps)

Run the cached-field loop using an explicit outgoing sparse cache for field
updates. This remains sparse and supports arbitrary recurrent topology.
"""
function run_outgoing_cache_probe!(context::C, nsteps::I) where {C<:FieldProbeContext,I<:Integer}
    spins = context.spins
    fields = context.fields
    active = context.active
    rng = context.rng
    checksum = zero(eltype(spins))

    if ORDER_MODE === :shuffle
        order = collect(eachindex(active))
        done = 0
        @inbounds while done < Int(nsteps)
            Random.shuffle!(rng, order)
            for active_pos in order
                done >= Int(nsteps) && break
                spin_idx = active[active_pos]
                field = fields[spin_idx]
                old_state = spins[spin_idx]
                new_state = proposal_state(old_state, field)
                delta = new_state - old_state
                spins[spin_idx] = new_state
                update_fields_after_spin_outgoing!(fields, context.out_colptr, context.out_idx, context.out_val, spin_idx, delta)
                checksum += fields[spin_idx]
                done += 1
            end
        end
        return checksum + sum(spins)
    end

    @inbounds for step_idx in 1:Int(nsteps)
        active_pos = ORDER_MODE === :cyclic ? mod1(step_idx, length(active)) : rand(rng, eachindex(active))
        spin_idx = active[active_pos]
        field = fields[spin_idx]
        old_state = spins[spin_idx]
        new_state = proposal_state(old_state, field)
        delta = new_state - old_state
        spins[spin_idx] = new_state
        update_fields_after_spin_outgoing!(fields, context.out_colptr, context.out_idx, context.out_val, spin_idx, delta)
        checksum += fields[spin_idx]
    end
    return checksum + sum(spins)
end

"""
    build_prototype(hidden)

Build one MNIST graph and one normalized MNIST input vector for the probe.
"""
function build_prototype(hidden::I) where {I<:Integer}
    graph = MNISTArchitecture(
        hidden = Int(hidden),
        output_replicas = OUTPUT_REPLICAS,
        precision = Float32,
        weight_scale = WEIGHT_SCALE,
        rng = Random.MersenneTwister(80_000 + Int(hidden)),
    )
    temp!(graph, TEMP)
    layer = MNISTLayer(graph = graph)
    x, _ = load_mnist_arrays(layer; split = :train, limit = 1)
    active = collect(vcat(InteractiveIsing.layerrange(graph[2]), InteractiveIsing.layerrange(graph[end])))
    nsteps = max(1, round(Int, SWEEPS * length(active)))
    return (; graph, x = copy(view(x, :, 1)), active, nsteps)
end

"""
    build_contexts(prototype, ncontexts; seed)

Create independent graph/state/field-cache contexts for one probe mode.
"""
function build_contexts(prototype::P, ncontexts::I; seed::S = 1) where {P,I<:Integer,S<:Integer}
    contexts = FieldProbeContext[]
    sizehint!(contexts, Int(ncontexts))
    for idx in 1:Int(ncontexts)
        graph = deepcopy(prototype.graph)
        temp!(graph, TEMP)
        IsingLearning.apply_input(graph, prototype.x)
        spins = InteractiveIsing.graphstate(graph)
        bias = graph_bias(graph)
        fields = similar(spins)
        sp = InteractiveIsing.adj(graph).sp
        compute_fields!(fields, sp, spins, bias)
        out_colptr, out_idx, out_val = sparse_outgoing_cache(sp)
        push!(contexts, FieldProbeContext(graph, spins, bias, fields, prototype.active, out_colptr, out_idx, out_val, Random.MersenneTwister(seed + idx)))
    end
    return contexts
end

"""
    timed_probe!(mode, context, nsteps)

Measure one context in either `:scan` or `:cache` mode.
"""
function timed_probe!(mode::M, context::C, nsteps::I) where {M<:Symbol,C<:FieldProbeContext,I<:Integer}
    result = Ref{Any}(nothing)
    seconds = @elapsed begin
        if mode === :scan
            result[] = run_scan_probe!(context, nsteps)
        elseif mode === :cache
            result[] = run_cache_probe!(context, nsteps)
        elseif mode === :outgoing_cache
            result[] = run_outgoing_cache_probe!(context, nsteps)
        else
            throw(ArgumentError("unknown probe mode $mode"))
        end
    end
    return (; seconds, result = result[])
end

"""
    run_parallel_probe!(mode, contexts, nsteps)

Spawn one task per context and return total wall time plus per-task timings.
"""
function run_parallel_probe!(mode::M, contexts::C, nsteps::I) where {M<:Symbol,C<:AbstractVector,I<:Integer}
    task_seconds = zeros(Float64, length(contexts))
    outputs = Vector{Any}(undef, length(contexts))
    total_seconds = if SCHEDULE_MODE === :spawn
        @elapsed begin
        tasks = map(eachindex(contexts)) do idx
            Threads.@spawn (idx = $idx, timed = timed_probe!($mode, $(contexts[idx]), $nsteps))
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
                timed = timed_probe!(mode, contexts[idx], nsteps)
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
    run_config(hidden, ncontexts, mode)

Run one field-cache probe configuration and append its timing row to CSV.
"""
function run_config(hidden::H, ncontexts::N, mode::M) where {H<:Integer,N<:Integer,M<:Symbol}
    prototype = build_prototype(hidden)

    single_context = only(build_contexts(prototype, 1; seed = 10_000))
    timed_probe!(mode, single_context, 1)
    single = timed_probe!(mode, single_context, prototype.nsteps)

    contexts = build_contexts(prototype, ncontexts; seed = 20_000)
    for context in contexts
        timed_probe!(mode, context, 1)
    end
    parallel = run_parallel_probe!(mode, contexts, prototype.nsteps)

    serial_all = single.seconds * Int(ncontexts)
    row = (;
        timestamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        hidden = Int(hidden),
        contexts = Int(ncontexts),
        threads = Threads.nthreads(),
        mode = string(mode),
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
    append_csv_row!(joinpath(OUTDIR, "mnist_field_cache_probe.csv"), row)
    println(row)
    flush(stdout)
    return row
end

"""
    main()

Compare sparse column scans with a cached-field update loop on MNIST graphs.
"""
function main()
    println(
        "MNIST field-cache probe hiddens=", HIDDENS,
        " workers=", WORKERS,
        " threads=", Threads.nthreads(),
        " order=", ORDER_MODE,
        " schedule=", SCHEDULE_MODE,
        " sweeps=", SWEEPS,
    )
    for hidden in HIDDENS
        for ncontexts in WORKERS
            run_config(hidden, ncontexts, :scan)
            run_config(hidden, ncontexts, :cache)
            run_config(hidden, ncontexts, :outgoing_cache)
        end
    end
    println("Saved outputs in ", OUTDIR)
end

main()
