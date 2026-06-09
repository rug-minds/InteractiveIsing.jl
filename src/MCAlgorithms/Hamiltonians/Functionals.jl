abstract type AbstractLinearFunctional end

struct H <: AbstractLinearFunctional end
struct ΔH <: AbstractLinearFunctional end


"""
Energy derivative w.r.t a single unit, i.e. ∂H/∂s_i
"""
struct d_iH <: AbstractLinearFunctional end

"""
The energy of a single unit
"""
struct H_i <: AbstractLinearFunctional end 

@inline function calculate(hF::AbstractLinearFunctional, hts::HTS, model, args...) where {HTS <: AbstractHamiltonianTerms}
    total = zero(eltype(model))
    total = @inline unrollreplace(total, hts...) do ftotal, hamiltonian
        ftotal = ftotal + @inline calculate(hF, hamiltonian, model, args...)
    end
    return total    
end

"""
    calculate!(dest, d_iH(), hamiltonian, model, proposal::MultiSpinProposal)

Write derivative components at the multi-spin proposal endpoint into `dest`.
Entry `i` is the derivative for `proposal.at_idxs[i]`.
"""
@inline function calculate!(
    dest,
    dh::d_iH,
    hamiltonian,
    model,
    proposal::MultiSpinProposal,
)
    spins = @inline graphstate(model)

    # Present the full endpoint while each derivative request identifies its
    # own target spin through a SingleSpinProposal.
    @inbounds for i in 1:length(proposal)
        spins[proposal.at_idxs[i]] = proposal.to_vals[i]
    end
    try
        @inbounds for i in 1:length(proposal)
            request = SingleSpinProposal{eltype(model)}(
                proposal.at_idxs[i],
                proposal.to_vals[i],
                NoChange(),
                proposal.layer_idxs[i],
                false,
            )
            dest[i] = @inline calculate(dh, hamiltonian, model, request)
        end
    finally
        @inbounds for i in 1:length(proposal)
            spins[proposal.at_idxs[i]] = proposal.from_vals[i]
        end
    end
    return dest
end

"""
    calculate(ΔH(), hts, model, proposal::MultiSpinProposal)

Fallback energy-change calculation for a multi-spin move on a collection of
Hamiltonian terms.

The proposal is decomposed into `FlipProposal`s and applied sequentially to the
model state while accumulating each single-spin `ΔH`. The original state is
restored before returning. This gives correct fallback semantics for terms that
only implement single-spin `ΔH`, while still allowing specialized
`MultiSpinProposal` methods to be added where the combined move can be computed
more directly.
"""
@inline function calculate(dh::ΔH, hts::HTS, model, proposal::MultiSpinProposal) where {HTS <: AbstractHamiltonianTerms}
    spins = @inline graphstate(model)
    total = zero(eltype(model))

    @inbounds for fp in subproposals(proposal)
        total += @inline calculate(dh, hts, model, fp)
        spins[at_idx(fp)] = to_val(fp)
    end

    @inbounds for i in 1:length(proposal)
        spins[proposal.at_idxs[i]] = proposal.from_vals[i]
    end

    return total
end

"""
    calculate(ΔH(), hterm, model, proposal::MultiSpinProposal)

Fallback energy-change calculation for one Hamiltonian term and a multi-spin
proposal.

The fallback walks through `subproposals(proposal)`, accumulates the single-spin
energy changes, temporarily applies accepted intermediate trial values to the
graph state, and restores the original spin values before returning.
"""
@inline function calculate(dh::ΔH, hterm::Hamiltonian, model, proposal::MultiSpinProposal)
    spins = @inline graphstate(model)
    total = zero(eltype(model))

    @inbounds for fp in subproposals(proposal)
        total += @inline calculate(dh, hterm, model, fp)
        spins[at_idx(fp)] = to_val(fp)
    end

    @inbounds for i in 1:length(proposal)
        spins[proposal.at_idxs[i]] = proposal.from_vals[i]
    end

    return total
end
