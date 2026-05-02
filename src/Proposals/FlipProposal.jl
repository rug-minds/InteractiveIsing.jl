export FlipProposal, MultiSpinProposal, to_delta_exp, to_delta_exp!, delta, accepteddelta, subproposals

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

function accept(proposer::AbstractProposer, f::FlipProposal)
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

"""
    MultiSpinProposal(at_idxs, from_vals, to_vals, layer_idxs[, accepted])

Proposal that changes several spins as one accepted/rejected move.

The four indexed fields must have the same length. Entry `i` represents the
single-spin change
`at_idxs[i] : from_vals[i] -> to_vals[i]` in `layer_idxs[i]`.

Use [`subproposals`](@ref) to view the proposal as a sequence of ordinary
`FlipProposal`s. Generic Hamiltonian fallbacks use that decomposition; term
implementations may add more efficient `MultiSpinProposal` methods when the
combined move can be handled directly.
"""
struct MultiSpinProposal{I,F,T,L} <: AbstractProposal
    at_idxs::I
    from_vals::F
    to_vals::T
    layer_idxs::L
    accepted::Bool

    function MultiSpinProposal{I,F,T,L}(at_idxs::I, from_vals::F, to_vals::T, layer_idxs::L, accepted::Bool) where {I,F,T,L}
        length(at_idxs) == length(from_vals) == length(to_vals) == length(layer_idxs) ||
            throw(ArgumentError("MultiSpinProposal fields must have matching lengths"))
        return new{I,F,T,L}(at_idxs, from_vals, to_vals, layer_idxs, accepted)
    end
end

function MultiSpinProposal(at_idxs::I, from_vals::F, to_vals::T, layer_idxs::L) where {I,F,T,L}
    return MultiSpinProposal{I,F,T,L}(at_idxs, from_vals, to_vals, layer_idxs, false)
end

function MultiSpinProposal(at_idxs::I, from_vals::F, to_vals::T, layer_idxs::L, accepted::Bool) where {I,F,T,L}
    return MultiSpinProposal{I,F,T,L}(at_idxs, from_vals, to_vals, layer_idxs, accepted)
end

isaccepted(r::MultiSpinProposal) = r.accepted
Base.length(r::MultiSpinProposal) = length(r.at_idxs)
Base.size(r::MultiSpinProposal) = (length(r),)
Base.eltype(r::MultiSpinProposal) = eltype(r.to_vals)
SparseArrays.rowvals(r::MultiSpinProposal) = r.at_idxs
SparseArrays.nonzeros(r::MultiSpinProposal) = r.to_vals

@inline at_idx(r::MultiSpinProposal, i::Int) = r.at_idxs[i]
@inline from_val(r::MultiSpinProposal, i::Int) = r.from_vals[i]
@inline to_val(r::MultiSpinProposal, i::Int) = r.to_vals[i]
@inline delta(r::MultiSpinProposal, i::Int) = r.to_vals[i] - r.from_vals[i]
@inline accepteddelta(r::MultiSpinProposal, i::Int) = r.accepted ? delta(r, i) : zero(eltype(r))

"""
    proposal[i]

Return the proposed value at graph index `i`, or zero when the multi-spin
proposal does not touch that index.
"""
@inline function Base.getindex(r::MultiSpinProposal, i::Int)
    idx = findfirst(==(i), r.at_idxs)
    return isnothing(idx) ? zero(eltype(r)) : r.to_vals[idx]
end

"""
    subproposal(proposal, i)

Return entry `i` of a `MultiSpinProposal` as a `FlipProposal` with the same
accepted flag.
"""
@inline function subproposal(r::MultiSpinProposal, i::Int)
    return FlipProposal(
        r.at_idxs[i],
        r.from_vals[i],
        r.to_vals[i],
        r.layer_idxs[i],
        r.accepted,
    )
end

struct MultiSpinSubProposals{P}
    proposal::P
end

"""
    subproposals(proposal::MultiSpinProposal)

Iterate over the single-spin `FlipProposal`s represented by `proposal`.

This is the canonical fallback decomposition for Hamiltonian `calculate` and
`update!` methods. The order is the order stored in `proposal.at_idxs`; callers
that rely on the fallback get sequential semantics in that order.
"""
subproposals(r::MultiSpinProposal) = MultiSpinSubProposals(r)
Base.length(s::MultiSpinSubProposals) = length(s.proposal)
Base.IteratorSize(::Type{<:MultiSpinSubProposals}) = Base.HasLength()
Base.eltype(::Type{<:MultiSpinSubProposals}) = FlipProposal

@inline function Base.iterate(s::MultiSpinSubProposals, state::Int = 1)
    state > length(s.proposal) && return nothing
    return subproposal(s.proposal, state), state + 1
end
