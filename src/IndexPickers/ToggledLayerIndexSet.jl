export ToggledLayerIndexSet, toggle!
"""
Toggle one layer on/off, and pick indices from the appropriate set
"""
struct ToggledLayerIndexSet{T} <: UniformIndexPicker
    set_with::Vector{T}
    set_without::Vector{T}
    on::Ref{Bool}
    sampling_indices_changed::Base.RefValue{Bool}
end

function ToggledLayerIndexSet(toggle_idx, layer_sets::UnitRange...)
    set_with = vcat((layer_sets[i] for i in toggle_idx)...)
    set_without = vcat((layer_sets[i] for i in 1:length(layer_sets) if i != toggle_idx)...)
    return ToggledLayerIndexSet(set_with, set_without, Ref(true), Ref(false))
end

function ToggledLayerIndexSet(g::IsingGraph, toggle_idx)
    layer_sets = layer_idxs(g)
    return ToggledLayerIndexSet(toggle_idx, layer_sets...)
end

function toggle!(is::ToggledLayerIndexSet)
    is.on[] = !is.on[]
    is.sampling_indices_changed[] = true
    return is
end

function on!(is::ToggledLayerIndexSet)
    if !is.on[]
        is.on[] = true
        is.sampling_indices_changed[] = true
    end
    return is
end

function off!(is::ToggledLayerIndexSet)
    if is.on[]
        is.on[] = false
        is.sampling_indices_changed[] = true
    end
    return is
end

@inline function sampling_indices(is::ToggledLayerIndexSet)
    return is.on[] ? is.set_with : is.set_without
end

@inline function sampling_changed!(is::ToggledLayerIndexSet)
    changed = is.sampling_indices_changed[]
    is.sampling_indices_changed[] = false
    return changed
end

function pick_idx(rng::R, is::ToggledLayerIndexSet) where {R <: AbstractRNG}
    if is.on[]
        return rand(rng, is.set_with)
    else
        return rand(rng, is.set_without)
    end
end
