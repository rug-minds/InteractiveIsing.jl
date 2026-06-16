export CoulombChargeShift

"""
    CoulombChargeShift(charge; split=0.5, charge_sign=nothing)

Defect mode for a mobile Coulomb free charge. Positive `charge` values use the
positive free-charge occupancy in `CoulombHamiltonian`; negative values use the
negative occupancy and store `abs(charge)` as the species magnitude.
"""
struct CoulombChargeShift{A,S,C} <: AbstractDefectMode
    charge::A
    split::S
    charge_sign::C
end

function CoulombChargeShift(charge::A; split::S = 0.5, charge_sign = nothing) where {A,S}
    selected_sign = isnothing(charge_sign) ? _defect_charge_sign(charge) : charge_sign
    return CoulombChargeShift{typeof(abs(charge)),S,typeof(selected_sign)}(abs(charge), split, selected_sign)
end

@inline _defect_charge_sign(charge) = charge < zero(charge) ? NegativeFreeCharge() : PositiveFreeCharge()

@inline _defect_convert_mode(mode::M, ::Type{T}) where {M<:CoulombChargeShift,T} =
    CoulombChargeShift(T(mode.charge); split = T(mode.split), charge_sign = mode.charge_sign)

"""
    _defect_has_coulomb_charge(effects)

Return whether a defect effect tuple contains a charged Coulomb mode.
"""
@inline _defect_has_coulomb_charge(::Tuple{}) = false
@inline _defect_has_coulomb_charge(effects::Tuple{<:CoulombChargeShift,Vararg}) = true
@inline _defect_has_coulomb_charge(effects::Tuple) = _defect_has_coulomb_charge(Base.tail(effects))

@inline _defect_charge_magnitude(c::CoulombHamiltonian, ::PositiveFreeCharge) = c.q_positive
@inline _defect_charge_magnitude(c::CoulombHamiltonian, ::NegativeFreeCharge) = c.q_negative

"""
    _defect_validate_mode!(mode::CoulombChargeShift, hterm, idx)

Validate that a charged-defect mode targets a Coulomb term, has a valid cell
split, and matches the Coulomb term's configured charge magnitude.
"""
function _defect_validate_mode!(mode::M, hterm::H, idx::I) where {M<:CoulombChargeShift,H<:CoulombHamiltonian,I<:Integer}
    zero(mode.split) <= mode.split <= one(mode.split) ||
        throw(ArgumentError("CoulombChargeShift split must lie in [0, 1]; got $(mode.split)."))
    expected = _defect_charge_magnitude(hterm, mode.charge_sign)
    isapprox(mode.charge, expected) ||
        throw(ArgumentError("CoulombChargeShift charge $(mode.charge) must match CoulombHamiltonian $(typeof(mode.charge_sign)) magnitude $expected. Configure q_positive/q_negative on CoulombHamiltonian."))
    isapprox(mode.split, hterm.free_charge_split) ||
        throw(ArgumentError("CoulombChargeShift split $(mode.split) must match CoulombHamiltonian free_charge_split $(hterm.free_charge_split)."))
    return true
end

"""
    _defect_coulomb_cell_coord(coulomb, graph, layer, graph_idx)

Map a defect graph index to the dipole cell occupied by that free charge.
"""
function _defect_coulomb_cell_coord(c::C, g::G, layer_arg::L, graph_idx::I) where {C<:CoulombHamiltonian,G<:AbstractIsingGraph,L,I<:Integer}
    layeridx(c) == Int(layer_arg) ||
        throw(ArgumentError("CoulombChargeShift targets CoulombHamiltonian layer $(layeridx(c)), but DefectHopping is bound to layer $(Int(layer_arg))."))

    layer = g[Int(layer_arg)]
    local_idx = Int(graph_idx) - Int(startidx(layer)) + 1
    1 <= local_idx <= nStates(layer) ||
        throw(BoundsError(layer, local_idx))

    return CartesianIndices(size(layer))[local_idx]
end

"""
    _defect_adjust_coulomb_occupancy!(mode, coulomb, graph, layer, idx, sign)

Add or remove one mobile free-charge occupancy represented by `mode`.
"""
function _defect_adjust_coulomb_occupancy!(mode::M, c::C, g::G, layer_arg::L, graph_idx::I, sign::S) where {M<:CoulombChargeShift,C<:CoulombHamiltonian,G<:AbstractIsingGraph,L,I<:Integer,S}
    coord = _defect_coulomb_cell_coord(c, g, layer_arg, graph_idx)
    if sign > 0
        add_cell_free_charge!(c, mode.charge_sign, coord)
    else
        remove_cell_free_charge!(c, mode.charge_sign, coord)
    end
    return nothing
end

@inline function _defect_adjust_coulomb_occupancy!(mode::M, c::C, g::G, layer_arg::L, graph_idx::I, sign::S) where {M<:AbstractDefectMode,C<:CoulombHamiltonian,G<:AbstractIsingGraph,L,I<:Integer,S}
    return nothing
end

"""
    _defect_apply_coulomb_effects!(coulomb, defects, idx, sign)

Apply all Coulomb free-charge occupancy changes carried by `defects`.
"""
function _defect_apply_coulomb_effects!(c::C, defects::D, graph_idx::I, sign::S) where {C<:CoulombHamiltonian,D<:DefectHopping,I<:Integer,S}
    _defect_has_coulomb_charge(defects.effects) || return nothing
    g = defects.state
    for mode in defects.effects
        _defect_adjust_coulomb_occupancy!(mode, c, g, defects.layer, graph_idx, sign)
    end
    return nothing
end

"""
    _defect_rebuild_coulomb!(coulomb, graph; validate=true)

Refresh derived Coulomb charge and potential after free-charge occupancy
changes.
"""
function _defect_rebuild_coulomb!(c::C, g::G; validate::Bool = true) where {C<:CoulombHamiltonian,G<:AbstractIsingGraph}
    rebuild_charge_density!(c, boundlayer(c, g); validate)
    recalc!(c)
    c.recalc_tracker[] = 1
    return c
end

"""
    _defect_apply_initial_mode!(mode::CoulombChargeShift, coulomb, graph, layer, idx, sign)

Register a charged defect in Coulomb free-charge occupancy during binding.
Neutrality is not checked here so separate positive and negative systems can be
bound before the final Coulomb initialization validates the total charge.
"""
function _defect_apply_initial_mode!(mode::M, c::C, g::G, layer_arg::L, graph_idx::I, sign::S) where {M<:CoulombChargeShift,C<:CoulombHamiltonian,G<:AbstractIsingGraph,L,I<:Integer,S}
    _defect_adjust_coulomb_occupancy!(mode, c, g, layer_arg, graph_idx, sign)
    _defect_rebuild_coulomb!(c, g; validate = false)
    return nothing
end

"""
    _defect_reapply_initialized_effects!(coulomb, defects)

No-op for occupancy-based Coulomb coupling because `CoulombHamiltonian.init!`
rebuilds `ρ` from its stored free-charge occupancy.
"""
function _defect_reapply_initialized_effects!(c::C, defects::D) where {C<:CoulombHamiltonian,D<:DefectHopping}
    return c
end

function _defect_reapply_initialized_effects!(hts::HTS, defects::D) where {HTS<:AbstractHamiltonianTerms,D<:DefectHopping}
    for hterm in hamiltonians(hts)
        _defect_reapply_initialized_effects!(hterm, defects)
    end
    return hts
end

"""
    _defect_coulomb_energy(coulomb)

Return the electrostatic energy represented by a Coulomb charge/potential pair.
"""
@inline function _defect_coulomb_energy(c::C) where {C<:CoulombHamiltonian}
    return eltype(c.ρ)(0.5) * sum(c.ρ .* c.u)
end

"""
    calculate(ΔH(), coulomb, defects, proposal::DefectHopProposal)

Calculate exact Coulomb energy change for a trial charged-defect hop by
temporarily moving free-charge occupancy, rebuilding `ρ`, and restoring all
Coulomb buffers.
"""
function calculate(::ΔH, c::C, defects::D, proposal::P) where {C<:CoulombHamiltonian,D<:DefectHopping,P<:DefectHopProposal}
    T = eltype(c.ρ)
    proposal.valid || return T(Inf)
    layeridx(c) == Int(defects.layer) || return zero(T)
    _defect_has_coulomb_charge(defects.effects) || return zero(T)

    ρ_before = copy(c.ρ)
    ρhat_before = copy(c.ρhat)
    uhat_before = copy(c.uhat)
    u_before = copy(c.u)
    pos_cell_before = copy(c.positive_cell_occupancy)
    neg_cell_before = copy(c.negative_cell_occupancy)
    pos_sheet_before = copy(c.positive_sheet_occupancy)
    neg_sheet_before = copy(c.negative_sheet_occupancy)

    recalc!(c)
    energy_before = _defect_coulomb_energy(c)

    _defect_apply_coulomb_effects!(c, defects, proposal.from_idx, -1)
    _defect_apply_coulomb_effects!(c, defects, proposal.to_idx, 1)
    _defect_rebuild_coulomb!(c, defects.state)
    energy_after = _defect_coulomb_energy(c)

    c.ρ .= ρ_before
    c.ρhat .= ρhat_before
    c.uhat .= uhat_before
    c.u .= u_before
    c.positive_cell_occupancy .= pos_cell_before
    c.negative_cell_occupancy .= neg_cell_before
    c.positive_sheet_occupancy .= pos_sheet_before
    c.negative_sheet_occupancy .= neg_sheet_before
    return energy_after - energy_before
end

"""
    update!(::Metropolis, coulomb, defects, proposal::DefectHopProposal)

Commit an accepted charged-defect hop to Coulomb free-charge occupancy and
refresh the derived charge and potential fields.
"""
function update!(algo::A, c::C, defects::D, proposal::P) where {A<:Metropolis,C<:CoulombHamiltonian,D<:DefectHopping,P<:DefectHopProposal}
    isaccepted(proposal) || return nothing
    proposal.valid || return nothing
    layeridx(c) == Int(defects.layer) || return nothing
    _defect_has_coulomb_charge(defects.effects) || return nothing

    _defect_apply_coulomb_effects!(c, defects, proposal.from_idx, -1)
    _defect_apply_coulomb_effects!(c, defects, proposal.to_idx, 1)
    _defect_rebuild_coulomb!(c, defects.state)
    return nothing
end
