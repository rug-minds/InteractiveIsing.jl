# AI Generated
export proposed_value

"""
    proposed_value(state, proposal)

Return the derivative target value for `proposal`. `NoChange()` endpoints read
the value from `state`; concrete endpoints are returned directly.
"""
@inline proposed_value(state, proposal::SingleSpinProposal) = to_val(proposal)

@inline function proposed_value(state, proposal::SingleSpinProposal{F,<:NoChange}) where {F}
    return @inbounds state[at_idx(proposal)]
end
