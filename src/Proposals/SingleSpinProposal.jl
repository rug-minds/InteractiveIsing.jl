# AI Generated
export SingleSpinProposal, FlipProposal, NoChange, delta, accepteddelta

"""
    NoChange()

Endpoint sentinel for a proposal request that should use the model's current
state at the proposed spin.
"""
struct NoChange end

"""
    SingleSpinProposal(at_idx, from_val, to_val, layer_idx[, accepted])

Proposal for one graph spin/state coordinate. `to_val` may be `NoChange()` for
derivative requests that should evaluate at the model's current state.
"""
struct SingleSpinProposal{F,T} <: AbstractProposal
    at_idx::Int
    from_val::F
    to_val::T
    layer_idx::Int
    accepted::Bool
end

@inline function SingleSpinProposal(at_idx, from_val::F, to_val::T, layer_idx, accept = false) where {F,T}
    return SingleSpinProposal{F,T}(Int(at_idx), from_val, to_val, Int(layer_idx), Bool(accept))
end

@inline function (::Type{SingleSpinProposal{F}})(at_idx, from_val, to_val, layer_idx, accept = false) where {F}
    endpoint = to_val isa NoChange ? to_val : F(to_val)
    return SingleSpinProposal{F,typeof(endpoint)}(Int(at_idx), F(from_val), endpoint, Int(layer_idx), Bool(accept))
end

"""
    SingleSpinProposal(proposal, at_idx, from_val, to_val, layer_idx[, accepted])

Build a new single-spin proposal while preserving the numeric endpoint type of
`proposal`.
"""
function SingleSpinProposal(
    proposal::SingleSpinProposal{F,T},
    at_idx,
    from_val,
    to_val,
    layer_idx,
    accept = false,
) where {F,T}
    endpoint = T <: NoChange ? NoChange() : T(to_val)
    return SingleSpinProposal{F,typeof(endpoint)}(Int(at_idx), F(from_val), endpoint, Int(layer_idx), Bool(accept))
end

const FlipProposal = SingleSpinProposal

"""
    delta(proposal::SingleSpinProposal)

Return the proposed endpoint displacement `to_val - from_val`.
"""
function delta(proposal::SingleSpinProposal)
    proposal.to_val isa NoChange && return zero(typeof(proposal.from_val))
    return proposal.to_val - proposal.from_val
end

"""
    accepteddelta(proposal::SingleSpinProposal)

Return the proposed displacement for accepted proposals and zero for rejected
proposals.
"""
function accepteddelta(proposal::SingleSpinProposal)
    return proposal.accepted ? delta(proposal) : zero(typeof(proposal.from_val))
end

"""
    accept(proposer, proposal::SingleSpinProposal)

Commit `proposal` to the proposer's state and return an accepted copy.
"""
function accept(proposer::AbstractProposer, proposal::SingleSpinProposal)
    to_val(proposal) isa NoChange &&
        throw(ArgumentError("Cannot accept a SingleSpinProposal whose endpoint is NoChange()."))
    spins = @inline InteractiveIsing.state(proposer.state)
    @inbounds spins[at_idx(proposal)] = to_val(proposal)
    return SingleSpinProposal(
        proposal,
        proposal.at_idx,
        proposal.from_val,
        proposal.to_val,
        proposal.layer_idx,
        true,
    )
end

isaccepted(proposal::SingleSpinProposal) = proposal.accepted

at_idx(proposal::SingleSpinProposal) = proposal.at_idx
from_val(proposal::SingleSpinProposal) = proposal.from_val
to_val(proposal::SingleSpinProposal) = proposal.to_val

Base.size(::SingleSpinProposal) = (1,)
Base.length(::SingleSpinProposal) = 1
Base.eltype(::SingleSpinProposal{F,T}) where {F,T} = F

SparseArrays.rowvals(proposal::SingleSpinProposal) = [proposal.at_idx]
SparseArrays.nonzeros(proposal::SingleSpinProposal) = [proposal.to_val]

"""
    proposal[i]

Return the proposed value at `i`, or zero when this single-spin proposal does
not touch `i`.
"""
@inline function Base.getindex(proposal::SingleSpinProposal, i::Int = proposal.at_idx)
    return i == proposal.at_idx && !(proposal.to_val isa NoChange) ? proposal.to_val : eltype(proposal)(0)
end
