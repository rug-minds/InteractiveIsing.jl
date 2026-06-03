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
    calculate(d_iH(), hts, model::AbstractVectorSpinGraph, spin_idx)

Accumulate the vector derivative for one spin in a vector-spin model.
"""
@inline function calculate(hF::d_iH, hts::HTS, model::G, args...) where {HTS<:AbstractHamiltonianTerms,G<:AbstractVectorSpinGraph}
    total = zero(spin_state_type(model))
    total = @inline unrollreplace(total, hts...) do ftotal, hamiltonian
        ftotal = ftotal + @inline calculate(hF, hamiltonian, model, args...)
    end
    return total
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
