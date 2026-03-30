export FlipProposal, to_delta_exp, to_delta_exp!, delta, accepteddelta

"""
rule to replace parameter :symb[i] with :((@view symb[]))
"""
struct FlipProposal{T} <: AbstractProposal
    at_idx::Int
    from_val::T
    to_val::T
    layer_idx::Int
    accepted::Bool
end

@inline function FlipProposal(at_idx, from_val::F, to_val::T, layer_idx, accept = false) where {F,T}
    return FlipProposal{T}(at_idx, from_val, T(to_val), layer_idx, accept)
end

function FlipProposal(fp::FlipProposal{T}, at_idx, from_val, to_val, layer_idx, accept = false) where {T}
    return FlipProposal{T}(at_idx, from_val, to_val, layer_idx, accept)
end

"""
Give the proposed delta
"""
function delta(fp::FlipProposal{T}) where {T}
    return fp.to_val - fp.from_val
end

"""
If the proposal is accepted, return the delta, otherwise return zero (which is the actual change to the system)
"""
function accepteddelta(fp::FlipProposal{T}) where {T}
    return fp.accepted ? delta(fp) : zero(T)
end

function accept(proposer::IsingGraphProposer, f::FlipProposal)
    spins = @inline InteractiveIsing.state(proposer.state)
    @inbounds spins[at_idx(f)] = to_val(f)
    FlipProposal(f, f.at_idx, f.from_val, f.to_val, f.layer_idx, true)
end
isaccepted(r::FlipProposal) = r.accepted

at_idx(r::FlipProposal) = r.at_idx
from_val(r::FlipProposal) = r.from_val
to_val(r::FlipProposal) = r.to_val

Base.size(r::FlipProposal) = (1,)
Base.length(r::FlipProposal) = 1
Base.eltype(r::FlipProposal{T}) where {T} = T

SparseArrays.rowvals(r::FlipProposal) = [r.at_idx]
SparseArrays.nonzeros(r::FlipProposal) = [r.to_val]
# SparseArrays.isassigned(r::FlipProposal) = true

# getsymb(r::Type{FlipProposal{S, Idx, T}}) where {S, Idx, T} = S
# getsymb(r::FlipProposal) = getsymb(typeof(r))
# getindex_symb(r::Type{FlipProposal{S, Idx, T}}) where {S, Idx, T} = Idx
# getindex_symb(r::FlipProposal) = getindex_symb(typeof(r))

@inline function Base.getindex(r::FlipProposal, i::Int = r.at_idx)
    i == r.at_idx ? r.to_val : eltype(r)(0)
end

Base.setindex!(r::FlipProposal, v, at_idx::Int) = begin r.at_idx = at_idx; r.to_val = v end

getidx(r::FlipProposal) = r.at_idx
getval(r::FlipProposal) = r.to_val
