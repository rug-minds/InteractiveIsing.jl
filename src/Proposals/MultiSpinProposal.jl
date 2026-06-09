# AI Generated
export MultiSpinProposal, subproposals

"""
    MultiSpinProposal(at_idxs, from_vals, to_vals, layer_idxs[, accepted])

Proposal that changes several spins as one accepted/rejected move.

The four indexed fields must have the same length. Entry `i` represents the
single-spin change `at_idxs[i] : from_vals[i] -> to_vals[i]` in `layer_idxs[i]`.
Use [`subproposals`](@ref) to view the proposal as a sequence of
`SingleSpinProposal`s.
"""
struct MultiSpinProposal{I,F,T,L} <: AbstractProposal
    at_idxs::I
    from_vals::F
    to_vals::T
    layer_idxs::L
    accepted::Bool

    function MultiSpinProposal{I,F,T,L}(
        at_idxs::I,
        from_vals::F,
        to_vals::T,
        layer_idxs::L,
        accepted::Bool,
    ) where {I,F,T,L}
        length(at_idxs) == length(from_vals) == length(to_vals) == length(layer_idxs) ||
            throw(ArgumentError("MultiSpinProposal fields must have matching lengths"))
        return new{I,F,T,L}(at_idxs, from_vals, to_vals, layer_idxs, accepted)
    end
end

function MultiSpinProposal(at_idxs::I, from_vals::F, to_vals::T, layer_idxs::L) where {I,F,T,L}
    return MultiSpinProposal{I,F,T,L}(at_idxs, from_vals, to_vals, layer_idxs, false)
end

function MultiSpinProposal(
    at_idxs::I,
    from_vals::F,
    to_vals::T,
    layer_idxs::L,
    accepted::Bool,
) where {I,F,T,L}
    return MultiSpinProposal{I,F,T,L}(at_idxs, from_vals, to_vals, layer_idxs, accepted)
end

isaccepted(proposal::MultiSpinProposal) = proposal.accepted
Base.length(proposal::MultiSpinProposal) = length(proposal.at_idxs)
Base.size(proposal::MultiSpinProposal) = (length(proposal),)
Base.eltype(proposal::MultiSpinProposal) = eltype(proposal.to_vals)
SparseArrays.rowvals(proposal::MultiSpinProposal) = proposal.at_idxs
SparseArrays.nonzeros(proposal::MultiSpinProposal) = proposal.to_vals

@inline at_idx(proposal::MultiSpinProposal, i::Int) = proposal.at_idxs[i]
@inline from_val(proposal::MultiSpinProposal, i::Int) = proposal.from_vals[i]
@inline to_val(proposal::MultiSpinProposal, i::Int) = proposal.to_vals[i]
@inline delta(proposal::MultiSpinProposal, i::Int) = proposal.to_vals[i] - proposal.from_vals[i]
@inline accepteddelta(proposal::MultiSpinProposal, i::Int) =
    proposal.accepted ? delta(proposal, i) : zero(eltype(proposal))

"""
    proposal[i]

Return the proposed value at graph index `i`, or zero when the multi-spin
proposal does not touch that index.
"""
@inline function Base.getindex(proposal::MultiSpinProposal, i::Int)
    idx = findfirst(==(i), proposal.at_idxs)
    return isnothing(idx) ? zero(eltype(proposal)) : proposal.to_vals[idx]
end

"""
    subproposal(proposal, i)

Return entry `i` of a `MultiSpinProposal` as a `SingleSpinProposal` with the
same accepted flag.
"""
@inline function subproposal(proposal::MultiSpinProposal, i::Int)
    return SingleSpinProposal(
        proposal.at_idxs[i],
        proposal.from_vals[i],
        proposal.to_vals[i],
        proposal.layer_idxs[i],
        proposal.accepted,
    )
end

struct MultiSpinSubProposals{P}
    proposal::P
end

"""
    subproposals(proposal::MultiSpinProposal)

Iterate over the single-spin proposals represented by `proposal`.
"""
subproposals(proposal::MultiSpinProposal) = MultiSpinSubProposals(proposal)
Base.length(subs::MultiSpinSubProposals) = length(subs.proposal)
Base.IteratorSize(::Type{<:MultiSpinSubProposals}) = Base.HasLength()
Base.eltype(::Type{<:MultiSpinSubProposals}) = SingleSpinProposal

@inline function Base.iterate(subs::MultiSpinSubProposals, state::Int = 1)
    state > length(subs.proposal) && return nothing
    return subproposal(subs.proposal, state), state + 1
end
