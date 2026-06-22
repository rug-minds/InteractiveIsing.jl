"""
    calculate(ΔH(), hamiltonian_terms, model, proposal::DefectHopProposal)

Calculate the defect-hop energy change over a Hamiltonian term collection.
"""
@inline function calculate(dh::ΔH, hts::HTS, model, proposal::P) where {HTS<:AbstractHamiltonianTerms,P<:DefectHopProposal}
    T = eltype(model)
    total = _defect_calculate_terms(dh, hamiltonians(hts), model, proposal, T)
    return total + _defect_hop_only_delta(model, proposal)
end

"""
    _defect_calculate_terms(ΔH(), terms, model, proposal, T)

Sum defect-hop energy terms through tuple recursion so each Hamiltonian term
keeps its concrete type during inference.
"""
@inline _defect_calculate_terms(dh::ΔH, terms::Tuple{}, model, proposal::P, ::Type{T}) where {P<:DefectHopProposal,T} = zero(T)

@inline function _defect_calculate_terms(
    dh::ΔH,
    terms::Tuple{H,Vararg},
    model,
    proposal::P,
    ::Type{T},
) where {H,P<:DefectHopProposal,T}
    return T(calculate(dh, first(terms), model, proposal)) +
        _defect_calculate_terms(dh, Base.tail(terms), model, proposal, T)
end

"""
    calculate(ΔH(), hamiltonian_terms, model::DefectsModel, proposal::ChargeHopProposal)

Route a combined charge proposal to the selected vacancy or carrier field.
"""
@inline function calculate(dh::ΔH, hts::HTS, model::M, proposal::P) where {HTS<:AbstractHamiltonianTerms,M<:DefectsModel,P<:ChargeHopProposal}
    return calculate(dh, hts, _charge_hop_system(model, proposal), proposal.proposal)
end

@inline function calculate(dh::ΔH, hterm::H, model::M, proposal::P) where {H<:Hamiltonian,M<:DefectsModel,P<:ChargeHopProposal}
    return calculate(dh, hterm, _charge_hop_system(model, proposal), proposal.proposal)
end

@inline function calculate(dh::ΔH, hterm::H, model::M, proposal::P) where {H<:HamiltonianTerm,M<:DefectsModel,P<:ChargeHopProposal}
    return calculate(dh, hterm, _charge_hop_system(model, proposal), proposal.proposal)
end

"""
    calculate(ΔH(), hterm::PolynomialHamiltonian, model, proposal::DefectHopProposal)

Calculate the polynomial local-potential energy change for one defect hop.
"""
@inline function calculate(::ΔH, hterm::H, model, proposal::P) where {H<:PolynomialHamiltonian,P<:DefectHopProposal}
    T = eltype(model)
    proposal.valid || return T(Inf)
    _defect_uses_localpotential(hterm.lp) || return zero(T)
    spins = @inline graphstate(model)
    from_state = @inbounds spins[proposal.from_idx]
    to_state = @inbounds spins[proposal.to_idx]
    total = zero(T)
    for mode in proposal.effects
        total += _defect_delta(mode, hterm, T, from_state, to_state, proposal.from_idx, proposal.to_idx)
    end
    return total
end

"""
    _defect_delta(mode, hterm, T, from_state, to_state)

Return one mode's energy contribution for a defect hop across a Hamiltonian
term.
"""
@inline function _defect_delta(mode::LocalPotentialShift{Order}, hterm::H, ::Type{T}, from_state, to_state) where {Order,H<:PolynomialHamiltonian{Order},T}
    return T(mode.hopping_scale) * T(hterm.c[]) * T(mode.strength) * (T(to_state)^Order - T(from_state)^Order)
end

@inline function _defect_delta(mode::LocalPotentialScaleCoupling{Order}, hterm::H, ::Type{T}, from_state, to_state, from_idx::I, to_idx::I) where {Order,H<:PolynomialHamiltonian{Order},T,I<:Integer}
    factor = T(mode.factor)
    from_lp = T(hterm.lp[Int(from_idx)])
    to_lp = T(hterm.lp[Int(to_idx)])
    from_delta = from_lp * (inv(factor) - one(T)) * T(from_state)^Order
    to_delta = to_lp * (factor - one(T)) * T(to_state)^Order
    return T(mode.hopping_scale) * T(hterm.c[]) * (from_delta + to_delta)
end

@inline _defect_delta(mode::AbstractDefectCoupling, hterm::H, ::Type{T}, from_state, to_state, from_idx::I, to_idx::I) where {H<:Hamiltonian,T,I<:Integer} =
    _defect_delta(mode, hterm, T, from_state, to_state)

@inline function _defect_delta(mode::ExtFieldShift, hterm::H, ::Type{T}, from_state, to_state) where {H<:ExtField,T}
    return -T(mode.hopping_scale) * T(hterm.c) * T(mode.strength) * (T(to_state) - T(from_state))
end

@inline _defect_delta(mode::AbstractDefectMode, hterm::Hamiltonian, ::Type{T}, from_state, to_state) where {T} = zero(T)

"""
    _defect_hop_only_delta(model, proposal)

Return energy terms that depend only on the proposed defect displacement, not
on any particular Hamiltonian storage field.
"""
@inline function _defect_hop_only_delta(model, proposal::P) where {P<:DefectHopProposal}
    T = eltype(model)
    proposal.valid || return T(Inf)
    total = zero(T)
    for mode in proposal.effects
        total += _defect_hop_only_delta(mode, proposal, T)
    end
    return total
end

@inline _defect_hop_only_delta(mode::AbstractDefectMode, proposal::P, ::Type{T}) where {P<:DefectHopProposal,T} = zero(T)

"""
    _defect_field_axis(mode, displacement)

Return the proposal axis used by an `ExtFieldChargeCoupling`.
"""
function _defect_field_axis(mode::M, displacement::D) where {M<:ExtFieldChargeCoupling,D<:Tuple}
    axis = isnothing(mode.axis) ? length(displacement) : Int(mode.axis)
    1 <= axis <= length(displacement) ||
        throw(ArgumentError("ExtFieldChargeCoupling axis $axis is outside proposal dimension $(length(displacement))."))
    return axis
end

"""
    _defect_extfield_charge_delta(mode, extfield, proposal, T)

Return the work done by the graph's external field during one charged hop.
"""
@inline function _defect_extfield_charge_delta(mode::M, hterm::H, proposal::P, ::Type{T}) where {M<:ExtFieldChargeCoupling,H<:ExtField,P<:DefectHopProposal,T}
    axis = _defect_field_axis(mode, proposal.displacement)
    field = T(hterm.c) * (T(hterm.b[proposal.from_idx]) + T(hterm.b[proposal.to_idx])) / T(2)
    return -T(mode.hopping_scale) * T(proposal.charge) * field * T(proposal.displacement[axis])
end

"""
    _defect_non_polynomial_delta(model, proposal)

Return the non-polynomial contribution to a defect-hop energy change.
"""
@inline function _defect_non_polynomial_delta(model, proposal::P) where {P<:DefectHopProposal}
    T = eltype(model)
    return proposal.valid ? zero(T) : T(Inf)
end

"""
    calculate(ΔH(), hterm, model, proposal::DefectHopProposal)

Return zero for non-polynomial terms, while preserving invalid-hop rejection.
"""
@inline calculate(::ΔH, hterm::H, model, proposal::P) where {H<:Bilinear,P<:DefectHopProposal} =
    _defect_non_polynomial_delta(model, proposal)

@inline function calculate(::ΔH, hterm::H, model, proposal::P) where {H<:ExtField,P<:DefectHopProposal}
    T = eltype(model)
    proposal.valid || return T(Inf)
    spins = @inline graphstate(model)
    from_state = @inbounds spins[proposal.from_idx]
    to_state = @inbounds spins[proposal.to_idx]
    total = zero(T)
    for mode in proposal.effects
        if mode isa ExtFieldShift
            _defect_uses_localpotential(hterm.b) &&
                (total += _defect_delta(mode, hterm, T, from_state, to_state))
        elseif mode isa ExtFieldChargeCoupling
            total += _defect_extfield_charge_delta(mode, hterm, proposal, T)
        end
    end
    return total
end

@inline calculate(::ΔH, hterm::H, model, proposal::P) where {H<:Clamping,P<:DefectHopProposal} =
    _defect_non_polynomial_delta(model, proposal)

@inline calculate(::ΔH, hterm::H, model, proposal::P) where {H<:CosineInteraction,P<:DefectHopProposal} =
    _defect_non_polynomial_delta(model, proposal)

@inline calculate(::ΔH, hterm::H, model, proposal::P) where {H<:GaussianBernoulli,P<:DefectHopProposal} =
    _defect_non_polynomial_delta(model, proposal)

@inline calculate(::ΔH, hterm::H, model, proposal::P) where {H<:SoftplusMarginNudging,P<:DefectHopProposal} =
    _defect_non_polynomial_delta(model, proposal)

@inline calculate(::ΔH, hterm::H, model, proposal::P) where {H<:LayerTerm,P<:DefectHopProposal} =
    _defect_non_polynomial_delta(model, proposal)

@inline calculate(::ΔH, hterm::H, model, proposal::P) where {H<:EmptyHamiltonian,P<:DefectHopProposal} =
    _defect_non_polynomial_delta(model, proposal)

"""
    update!(::Metropolis, hterm::PolynomialHamiltonian, model, proposal::DefectHopProposal)

Move the accepted defect charge between polynomial local-potential entries.
"""
function update!(algo::A, hterm::H, model, proposal::P) where {A<:Metropolis,H<:PolynomialHamiltonian,P<:DefectHopProposal}
    isaccepted(proposal) || return nothing
    proposal.valid || return nothing

    _defect_apply_effects!(hterm, proposal.effects, proposal.from_idx, -1)
    _defect_apply_effects!(hterm, proposal.effects, proposal.to_idx, 1)
    return nothing
end

"""
    update!(::Metropolis, hterm::ExtField, model, proposal::DefectHopProposal)

Move accepted defect field shifts between magnetic-field entries.
"""
function update!(algo::A, hterm::H, model, proposal::P) where {A<:Metropolis,H<:ExtField,P<:DefectHopProposal}
    isaccepted(proposal) || return nothing
    proposal.valid || return nothing

    _defect_apply_effects!(hterm, proposal.effects, proposal.from_idx, -1)
    _defect_apply_effects!(hterm, proposal.effects, proposal.to_idx, 1)
    return nothing
end
