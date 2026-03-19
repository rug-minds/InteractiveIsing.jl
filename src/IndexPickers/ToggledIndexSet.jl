export ToggledIndexSet, toggle
"""
Structs that pick indices from a set with Uniform probability
"""
struct ToggledIndexSet{T,I} <: UniformIndexPicker
    layer_ranges::T
    on::Vector{Bool}
    cum_lengths::Vector{I} #Start at 0
end

function ToggledIndexSet(layer_ranges...)
    cum_lengths = append!([eltype(layer_ranges[1])(0)], cumsum(length.(layer_ranges[1:end])))
    on = [true for _ in layer_ranges]
    return ToggledIndexSet(layer_ranges, on, cum_lengths)
end

function ToggledIndexSet(g::IsingGraph)
    layer_ranges = layer_idxs(g)
    return ToggledIndexSet(layer_ranges...)
end

function Base.length(is::ToggledIndexSet, layer_idx::Int)
    length(is.layer_ranges[layer_idx])
end

function cumulative_length(is::ToggledIndexSet, layer_idx::Int)
    is.cum_lengths[layer_idx]
end

function cumulative_length(is::ToggledIndexSet)
    is.cum_lengths[end]
end

function toggle(is::ToggledIndexSet, layer_idx::Int)
    step_func(b::Bool) = b ? 1 : -1
    is.on[layer_idx] = !is.on[layer_idx]
    # Fix the cumulative lengths
    on = is.on[layer_idx]
    this_length = length(is, layer_idx)
    cum_lengths = is.cum_lengths
    # Fix the cumulative lengths for all layers after the toggled one
    cum_lengths[layer_idx+1:end] .+= step_func(on) * this_length
    # Fix the total length
    return is
end

function pick_idx(rng::R, is::ToggledIndexSet) where {R <: AbstractRNG}
    # Binary search on the cumulative lengths
    # Find the last index where idx <= cum_lengths[idx]
    el_idx = rand(rng, 1:cumulative_length(is))
    layer_idx = searchsortedfirst(is.cum_lengths, el_idx) - 1
    # Find the local index within the layer
    cum_length_before = cumulative_length(is, layer_idx)
    local_idx = el_idx - cum_length_before
    return is.layer_ranges[layer_idx][local_idx]
end
