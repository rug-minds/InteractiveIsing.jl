export LayerwiseMC, LayerAlgorithm, LayerScheduler, SequentialLayerScheduler

abstract type LayerScheduler end

"""
    SequentialLayerScheduler(counts = ())

Minimal deterministic scheduler for [`LayerwiseMC`](@ref).

`counts[i]` is the number of subalgorithm steps run for the `i`th
`LayerAlgorithm` in one composite step. Empty `counts` means one step per
subalgorithm and is resolved when the composite algorithm is constructed.
"""
struct SequentialLayerScheduler{Counts<:Tuple} <: LayerScheduler
    counts::Counts
end

SequentialLayerScheduler() = SequentialLayerScheduler(())

"""
    LayerAlgorithm(layer, algorithm)

One layer-scoped Monte Carlo subalgorithm used by [`LayerwiseMC`](@ref).
"""
struct LayerAlgorithm{A}
    layer::Int
    algorithm::A
end

LayerAlgorithm(layer::Integer, algorithm) = LayerAlgorithm(Int(layer), algorithm)

"""
    LayerwiseMC(specs...)
    LayerwiseMC(1 => LocalLangevin(...), 2 => Metropolis();
                scheduler = SequentialLayerScheduler((1, 1)))

Composite Monte Carlo algorithm that runs arbitrary existing MC algorithms on
specific graph layers.

Each subalgorithm is initialized on the original graph while `model.index_set`
is temporarily restricted to the selected layer. Stepping then uses the cached
subcontext directly. Layer ordering and repeat counts are owned by the scheduler.
"""
struct LayerwiseMC{Specs<:Tuple,Scheduler<:LayerScheduler} <: IsingMCAlgorithm
    specs::Specs
    scheduler::Scheduler
end

function _layerwise_resolve_scheduler(scheduler::SequentialLayerScheduler{Tuple{}}, n::Integer)
    return SequentialLayerScheduler(ntuple(_ -> 1, n))
end

function _layerwise_resolve_scheduler(scheduler::SequentialLayerScheduler, n::Integer)
    length(scheduler.counts) == n ||
        throw(ArgumentError("SequentialLayerScheduler count length $(length(scheduler.counts)) does not match number of layer algorithms $(n)."))
    return SequentialLayerScheduler(map(count -> max(1, Int(count)), scheduler.counts))
end

function LayerwiseMC(specs::LayerAlgorithm...; scheduler::LayerScheduler = SequentialLayerScheduler())
    return LayerwiseMC(specs, _layerwise_resolve_scheduler(scheduler, length(specs)))
end

function LayerwiseMC(pairs::Pair...; scheduler::LayerScheduler = SequentialLayerScheduler())
    specs = ntuple(length(pairs)) do i
        pair = pairs[i]
        LayerAlgorithm(first(pair), last(pair))
    end
    return LayerwiseMC(specs...; scheduler)
end

@inline _layerwise_layer_range(model, layer_idx::Int) =
    inline_layer_dispatch(layer -> graphidxs(layer), layer_idx, layers(model))

@inline _layerwise_spec_indices(specs::Tuple) = ntuple(i -> Val(i), length(specs))

@inline init_layer_scheduler(::LayerScheduler, algorithm::LayerwiseMC, context) = nothing

@generated function _layerwise_context_updates(subcontext::S, out::O) where {S<:NamedTuple,O<:NamedTuple}
    s_names = S.parameters[1]
    s_types = S.parameters[2].parameters
    o_names = O.parameters[1]
    o_types = O.parameters[2].parameters

    kept = Symbol[]
    for (out_pos, name) in enumerate(o_names)
        sub_pos = findfirst(==(name), s_names)
        isnothing(sub_pos) && continue

        old_type = s_types[sub_pos]
        new_type = o_types[out_pos]
        old_type <: Base.RefValue && !(new_type <: Base.RefValue) && continue
        push!(kept, name)
    end

    kept_tuple = Tuple(kept)
    values = [:(getfield(out, $(QuoteNode(name)))) for name in kept]
    return :(NamedTuple{$kept_tuple}(($(values...),)))
end

@inline _layerwise_update_context(subcontext, out) =
    merge(subcontext, _layerwise_context_updates(subcontext, out))

@inline function _layerwise_init_spec(spec::LayerAlgorithm, context)
    model = context.model
    original_index_set = getfield(model, :index_set)
    layer_index_set = _layerwise_layer_range(model, spec.layer)
    setfield!(model, :index_set, layer_index_set)
    subcontext = Processes.init(spec.algorithm, context)
    setfield!(model, :index_set, original_index_set)
    return subcontext
end

function Processes.init(algorithm::LayerwiseMC, context)
    (; model) = context

    layer_index_sets = map(spec -> _layerwise_layer_range(model, spec.layer), algorithm.specs)
    subcontexts = map(spec -> Ref(_layerwise_init_spec(spec, context)), algorithm.specs)
    scheduler_context = init_layer_scheduler(algorithm.scheduler, algorithm, context)
    spec_indices = _layerwise_spec_indices(algorithm.specs)

    return (;
        model,
        layer_index_sets,
        subcontexts,
        scheduler_context,
        spec_indices,
    )
end

@inline _layerwise_accepted(out) =
    get(out, :accepted, hasproperty(out, :proposal) && isaccepted(out.proposal) ? 1 : 0)

@inline _layerwise_attempted(out) =
    get(out, :attempted, hasproperty(out, :proposal) ? 1 : 0)

@inline function _layerwise_step_spec!(algorithm::LayerwiseMC, context, spec_idx::Int)
    return inline_layer_dispatch(static_idx -> _layerwise_step_spec!(algorithm, context, static_idx), spec_idx, context.spec_indices)
end

@inline function _layerwise_step_spec!(algorithm::LayerwiseMC, context, ::Val{I}) where {I}
    spec = algorithm.specs[I]
    subcontext_ref = context.subcontexts[I]
    subcontext = subcontext_ref[]
    out = Processes.step!(spec.algorithm, subcontext)
    subcontext_ref[] = _layerwise_update_context(subcontext, out)
    return out
end

function _layerwise_scheduler_step!(scheduler::SequentialLayerScheduler, algorithm::LayerwiseMC, context)
    accepted = 0
    attempted = 0

    for spec_idx in eachindex(scheduler.counts)
        for _ in 1:scheduler.counts[spec_idx]
            out = _layerwise_step_spec!(algorithm, context, spec_idx)

            accepted += _layerwise_accepted(out)
            attempted += _layerwise_attempted(out)
        end
    end

    acceptance_rate = attempted == 0 ? zero(eltype(context.model)) : eltype(context.model)(accepted) / eltype(context.model)(attempted)
    return (; accepted, attempted, acceptance_rate)
end

Processes.step!(algorithm::LayerwiseMC, context) =
    _layerwise_scheduler_step!(algorithm.scheduler, algorithm, context)
