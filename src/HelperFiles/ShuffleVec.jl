"""
Vector that allows the shuffling of indexes without moving any internal data
"""
struct ShuffleVec{T} <: AbstractVector{T}
    data::Vector{T}
    idxs::Vector{Int}

    function ShuffleVec(data::Vector{T}) where T
        idxs = collect(1:length(data))
        return new{T}(data, idxs)
    end

    function ShuffleVec{T}() where T
        return new{T}(Vector{T}(), Vector{Int}())
    end
end

Base.size(sv::ShuffleVec) = size(sv.data)
Base.getindex(sv::ShuffleVec, i::Int) = sv.data[sv.idxs[i]]
Base.setindex!(sv::ShuffleVec, val, i::Int) = sv.data[sv.idxs[i]] = val
Base.IndexStyle(::Type{<:ShuffleVec}) = IndexLinear()
Base.eltype(p::ShuffleVec{T}) where T = T
Base.length(p::ShuffleVec) = length(p.data)
Base.iterate(p::ShuffleVec, state = 1) = state > length(p) ? nothing : (p[state], state+1)
unshuffled(p::ShuffleVec) = p.data

function Base.deleteat!(p::ShuffleVec, i::Integer, datafunc::Function = (vars...) -> nothing)
    internal_idx = p.idxs[i]
    deleteat!(p.data, p.idxs[i])
    deleteat!(p.idxs, i)

    # Update idxs
    for idx in eachindex(p.idxs)
        if p.idxs[idx] > internal_idx
            p.idxs[idx] -= 1
        end
    end

    # Cleanup internal data for all elements that have been shifted
    for new_i_idx in internal_idx:length(p)
        data_el = p.data[new_i_idx]
        datafunc(data_el, new_i_idx)
    end

    return p
end

# For do syntax
Base.deleteat!(f::Function, p::ShuffleVec, i::Integer) = deleteat!(p, i, f)

function Base.deleteat!(p::ShuffleVec, i::AbstractVector)
    for idx in i
        deleteat!(p.data, p.idxs[idx])
    end
    return p
end

function Base.push!(p::ShuffleVec, items...)
    push!(p.data, items...)
    push!(p.idxs, length(p.data))
    return p
end

function shuffle!(p::ShuffleVec, oldidx, newidx)
    newidxs = copy(p.idxs)
    internal_idx = p.idxs[oldidx]

    shift_right = newidx < oldidx
    block = shift_right ? (newidx:(oldidx-1)) : ((oldidx+1):newidx)

    if shift_right
        newidxs[block.+1] = p.idxs[block]
        newidxs[newidx] = internal_idx        
    else
        newidxs[block.-1] = p.idxs[block]
        newidxs[newidx] = internal_idx
    end

    p.idxs .= newidxs
end

@inline internalidx(p::ShuffleVec, external_idx::Integer) = p.idxs[external_idx]
@inline externalidx(p::ShuffleVec, internal_idx::Integer) = findfirst(p.idxs .== internal_idx)

Base.convert(::Type{ShuffleVec{T}}, p::Vector{T}) where T = ShuffleVec(p)
Base.convert(::Type{Vector{T}}, p::ShuffleVec{T}) where T = p.data