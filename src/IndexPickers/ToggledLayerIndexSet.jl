export ToggledLayerIndexSet, toggle!
"""
Toggle one layer on/off, and pick indices from the appropriate set
"""
struct ToggledLayerIndexSet{T} <: UniformIndexPicker
    set_with::Vector{T}
    set_without::Vector{T}
    on::Ref{Bool}
end

function ToggledLayerIndexSet(toggle_idx, layer_sets::UnitRange...)
    set_with = vcat((layer_sets[i] for i in toggle_idx)...)
    set_without = vcat((layer_sets[i] for i in 1:length(layer_sets) if i != toggle_idx)...)
    return ToggledLayerIndexSet(set_with, set_without, Ref(true))
end

function ToggledLayerIndexSet(g::IsingGraph, toggle_idx)
    layer_sets = layer_idxs(g)
    return ToggledLayerIndexSet(toggle_idx, layer_sets...)
end

function toggle!(is::ToggledLayerIndexSet)
    is.on[] = !is.on[]
    return is
end

function on!(is::ToggledLayerIndexSet)
    is.on[] = true
    return is
end

function off!(is::ToggledLayerIndexSet)
    is.on[] = false
    return is
end

function pick_idx(rng::R, is::ToggledLayerIndexSet) where {R <: AbstractRNG}
    if is.on[]
        return rand(rng, is.set_with)
    else
        return rand(rng, is.set_without)
    end
end


