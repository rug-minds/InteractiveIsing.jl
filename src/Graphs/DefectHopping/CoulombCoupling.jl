export CoulombChargeCoupling, CoulombChargeShift

"""
    CoulombChargeCoupling(charge; split=0.5, charge_sign=nothing)

Mobile-species coupling hamiltonian for Coulomb free charge. Positive `charge`
values use the positive free-charge occupancy in `CoulombHamiltonian`; negative
values use the negative occupancy and store `abs(charge)` as the species
magnitude.
"""
struct CoulombChargeCoupling{A,S,C,K} <: AbstractDefectCoupling
    charge::A
    split::S
    charge_sign::C
    kernel::K
end

function CoulombChargeCoupling(charge::A; split::S = 0.5, charge_sign = nothing, kernel = nothing) where {A,S}
    selected_sign = isnothing(charge_sign) ? _defect_charge_sign(charge) : charge_sign
    return CoulombChargeCoupling{typeof(abs(charge)),S,typeof(selected_sign),typeof(kernel)}(abs(charge), split, selected_sign, kernel)
end

const CoulombChargeShift = CoulombChargeCoupling

@inline _defect_charge_sign(charge) = charge < zero(charge) ? NegativeFreeCharge() : PositiveFreeCharge()

@inline function _defect_convert_mode(mode::M, g::G) where {M<:CoulombChargeCoupling,G<:AbstractIsingGraph}
    charge = _defect_internal_value(mode.charge, physicalunits(charge = 1, role = :free_charge), g, :coulomb_charge_shift)
    split = _defect_internal_value(mode.split, physicalunits(role = :dimensionless), g, :free_charge_split)
    return CoulombChargeCoupling(charge; split, charge_sign = mode.charge_sign)
end

@inline _defect_mode_uses_physical_charge(::CoulombChargeCoupling) = true

"""
    _defect_has_coulomb_charge(effects)

Return whether a defect effect tuple contains a charged Coulomb mode.
"""
@inline _defect_has_coulomb_charge(::Tuple{}) = false
@inline _defect_has_coulomb_charge(effects::Tuple{<:CoulombChargeCoupling,Vararg}) = true
@inline _defect_has_coulomb_charge(effects::Tuple) = _defect_has_coulomb_charge(Base.tail(effects))

@inline _defect_charge_magnitude(c::CoulombHamiltonian, ::PositiveFreeCharge) = c.q_positive
@inline _defect_charge_magnitude(c::CoulombHamiltonian, ::NegativeFreeCharge) = c.q_negative

"""
    _defect_validate_mode!(mode::CoulombChargeCoupling, hterm, idx)

Validate that a charged-defect mode targets a Coulomb term, has a valid cell
split, and matches the Coulomb term's configured charge magnitude.
"""
function _defect_validate_mode!(mode::M, hterm::H, idx::I) where {M<:CoulombChargeCoupling,H<:CoulombHamiltonian,I<:Integer}
    zero(mode.split) <= mode.split <= one(mode.split) ||
        throw(ArgumentError("CoulombChargeCoupling split must lie in [0, 1]; got $(mode.split)."))
    expected = _defect_charge_magnitude(hterm, mode.charge_sign)
    isapprox(mode.charge, expected) ||
        throw(ArgumentError("CoulombChargeCoupling charge $(mode.charge) must match CoulombHamiltonian $(typeof(mode.charge_sign)) magnitude $expected. Configure q_positive/q_negative on CoulombHamiltonian."))
    isapprox(mode.split, hterm.free_charge_split) ||
        throw(ArgumentError("CoulombChargeCoupling split $(mode.split) must match CoulombHamiltonian free_charge_split $(hterm.free_charge_split)."))
    return true
end

"""
    _defect_coulomb_cell_coord(coulomb, graph, layer, graph_idx)

Map a defect graph index to the dipole cell occupied by that free charge.
"""
function _defect_coulomb_cell_coord(c::C, g::G, layer_arg::L, graph_idx::I) where {C<:CoulombHamiltonian,G<:AbstractIsingGraph,L,I<:Integer}
    layeridx(c) == Int(layer_arg) ||
        throw(ArgumentError("CoulombChargeCoupling targets CoulombHamiltonian layer $(layeridx(c)), but DefectHopping is bound to layer $(Int(layer_arg))."))

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
function _defect_adjust_coulomb_occupancy!(mode::M, c::C, g::G, layer_arg::L, graph_idx::I, sign::S) where {M<:CoulombChargeCoupling,C<:CoulombHamiltonian,G<:AbstractIsingGraph,L,I<:Integer,S}
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

struct _DefectCoulombSheetDelta{T}
    coord::CartesianIndex{3}
    charge::T
end

const _DEFECT_COULOMB_KERNEL_CACHE = IdDict{Any,Any}()

"""
    _defect_precompute_free_charge_kernel(coulomb)

Build the sheet-charge Green response used by Coulomb defect hops. The cache is
owned by the defect coupling layer because ordinary Coulomb spin dynamics does
not need this proposal kernel.
"""
function _defect_precompute_free_charge_kernel(c::C) where {C<:CoulombHamiltonian}
    T = eltype(c.ρ)
    Nx, Ny, Nz = size(c.ρ)
    Nxh = size(c.ρhat, 1)
    invden = c.inv_den
    m_upper = c.mod_upperd
    scale = -(c.az^2) / (c.ϵ * T(Nx * Ny))

    kernel = zeros(T, Nx, Ny, Nz, Nz)
    ρtmp = zeros(T, Nx, Ny, Nz)
    ρhat_tmp = similar(c.ρhat)
    uhat_tmp = similar(c.uhat)
    utmp = zeros(T, Nx, Ny, Nz)
    dptmp = similar(c.dp_scratch)

    @inbounds for source_z in 1:Nz
        ρtmp[1, 1, source_z] = one(T)
        mul!(ρhat_tmp, c.Pxy, ρtmp)

        for ny in 1:Ny, nx in 1:Nxh
            if !isfinite(invden[nx, ny, 1]) || !isfinite(invden[nx, ny, Nz])
                uhat_tmp[nx, ny, :] .= zero(Complex{T})
                continue
            end

            dptmp[nx, ny, 1] = (scale * ρhat_tmp[nx, ny, 1]) * invden[nx, ny, 1]
            for nz in 2:Nz
                dptmp[nx, ny, nz] = (scale * ρhat_tmp[nx, ny, nz] - dptmp[nx, ny, nz - 1]) * invden[nx, ny, nz]
            end

            uhat_tmp[nx, ny, Nz] = dptmp[nx, ny, Nz]
            for nz in (Nz - 1):-1:1
                uhat_tmp[nx, ny, nz] = dptmp[nx, ny, nz] - m_upper[nx, ny, nz] * uhat_tmp[nx, ny, nz + 1]
            end
        end

        mul!(utmp, c.iPxy, uhat_tmp)
        kernel[:, :, :, source_z] .= utmp
        ρtmp[1, 1, source_z] = zero(T)
    end

    return kernel
end

"""
    _defect_free_charge_kernel(coulomb)

Return the cached free-charge Green kernel for a Coulomb Hamiltonian internal
state, computing it once on first use by defect hopping.
"""
function _defect_free_charge_kernel(c::C) where {C<:CoulombHamiltonian}
    T = eltype(c.ρ)
    key = getfield(c, :internal)
    kernel = get!(_DEFECT_COULOMB_KERNEL_CACHE, key) do
        _defect_precompute_free_charge_kernel(c)
    end
    return kernel::Array{T,4}
end

@inline _defect_mode_kernel(mode::CoulombChargeCoupling{A,S,C,Nothing}, c::CoulombHamiltonian) where {A,S,C} =
    _defect_free_charge_kernel(c)

@inline _defect_mode_kernel(mode::CoulombChargeCoupling{A,S,C,K}, c::CoulombHamiltonian) where {A,S,C,K<:Array} =
    mode.kernel

"""
    _defect_bind_mode(mode::CoulombChargeCoupling, coulomb, graph, layer)

Attach the Coulomb free-charge Green kernel to a bound defect mode. The kernel
remains owned by this coupling extension, but the hot hop path can read it
directly from the mode without a cache lookup.
"""
function _defect_bind_mode(mode::M, c::C, g::G, layer_arg::L) where {M<:CoulombChargeCoupling,C<:CoulombHamiltonian,G<:AbstractIsingGraph,L}
    return CoulombChargeCoupling(
        mode.charge;
        split = mode.split,
        charge_sign = mode.charge_sign,
        kernel = _defect_free_charge_kernel(c),
    )
end

"""
    _defect_free_charge_green(coulomb, kernel, target, source)

Return the potential at `target` from a unit free sheet charge at `source`.
"""
@inline function _defect_free_charge_green(c::C, kernel::Array{T,4}, target::CartesianIndex{3}, source::CartesianIndex{3}) where {C<:CoulombHamiltonian,T}
    Nx, Ny, _ = size(c.ρ)
    dx = mod1(target[1] - source[1] + 1, Nx)
    dy = mod1(target[2] - source[2] + 1, Ny)
    return @inbounds kernel[dx, dy, target[3], source[3]]
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
    _defect_apply_initial_mode!(mode::CoulombChargeCoupling, coulomb, graph, layer, idx, sign)

Register a charged defect in Coulomb free-charge occupancy during binding.
Neutrality is not checked here so separate positive and negative systems can be
bound before the final Coulomb initialization validates the total charge.
"""
function _defect_apply_initial_mode!(mode::M, c::C, g::G, layer_arg::L, graph_idx::I, sign::S) where {M<:CoulombChargeCoupling,C<:CoulombHamiltonian,G<:AbstractIsingGraph,L,I<:Integer,S}
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
    _defect_append_coulomb_deltas!(deltas, mode, coulomb, defects, proposal)

Append the sheet-charge changes caused by one cell-centered charged-defect hop.
"""
function _defect_append_coulomb_deltas!(
    deltas::Vector{_DefectCoulombSheetDelta{T}},
    mode::M,
    c::C,
    defects::D,
    proposal::P,
) where {T,M<:CoulombChargeCoupling,C<:CoulombHamiltonian,D<:DefectHopping,P<:DefectHopProposal}
    g = defects.state
    from_cell = _defect_coulomb_cell_coord(c, g, defects.layer, proposal.from_idx)
    to_cell = _defect_coulomb_cell_coord(c, g, defects.layer, proposal.to_idx)
    q = T(_free_charge_sign(mode.charge_sign)) * T(mode.charge)
    split = T(mode.split)
    lower_weight = one(T) - split

    push!(deltas, _DefectCoulombSheetDelta(CartesianIndex(from_cell[1], from_cell[2], from_cell[3]), -q * lower_weight))
    push!(deltas, _DefectCoulombSheetDelta(CartesianIndex(from_cell[1], from_cell[2], from_cell[3] + 1), -q * split))
    push!(deltas, _DefectCoulombSheetDelta(CartesianIndex(to_cell[1], to_cell[2], to_cell[3]), q * lower_weight))
    push!(deltas, _DefectCoulombSheetDelta(CartesianIndex(to_cell[1], to_cell[2], to_cell[3] + 1), q * split))
    return deltas
end

"""
    _defect_coulomb_mode_deltas(mode, coulomb, defects, proposal, T)

Return the four sheet-charge deltas generated by one cell-centered Coulomb
charge hop.
"""
@inline function _defect_coulomb_mode_deltas(
    mode::M,
    c::C,
    defects::D,
    proposal::P,
    ::Type{T},
) where {M<:CoulombChargeCoupling,C<:CoulombHamiltonian,D<:DefectHopping,P<:DefectHopProposal,T}
    g = defects.state
    from_cell = _defect_coulomb_cell_coord(c, g, defects.layer, proposal.from_idx)
    to_cell = _defect_coulomb_cell_coord(c, g, defects.layer, proposal.to_idx)
    q = T(_free_charge_sign(mode.charge_sign)) * T(mode.charge)
    split = T(mode.split)
    lower_weight = one(T) - split

    return (
        _DefectCoulombSheetDelta(CartesianIndex(from_cell[1], from_cell[2], from_cell[3]), -q * lower_weight),
        _DefectCoulombSheetDelta(CartesianIndex(from_cell[1], from_cell[2], from_cell[3] + 1), -q * split),
        _DefectCoulombSheetDelta(CartesianIndex(to_cell[1], to_cell[2], to_cell[3]), q * lower_weight),
        _DefectCoulombSheetDelta(CartesianIndex(to_cell[1], to_cell[2], to_cell[3] + 1), q * split),
    )
end

@inline function _defect_append_coulomb_deltas!(
    deltas::Vector{_DefectCoulombSheetDelta{T}},
    mode::M,
    c::C,
    defects::D,
    proposal::P,
) where {T,M<:AbstractDefectMode,C<:CoulombHamiltonian,D<:DefectHopping,P<:DefectHopProposal}
    return deltas
end

"""
    _defect_coulomb_deltas(coulomb, defects, proposal, T)

Return the sheet-charge delta vector for a proposed charged-defect hop.
"""
function _defect_coulomb_deltas(c::C, defects::D, proposal::P, ::Type{T}) where {C<:CoulombHamiltonian,D<:DefectHopping,P<:DefectHopProposal,T}
    deltas = _DefectCoulombSheetDelta{T}[]
    sizehint!(deltas, 4)
    for mode in defects.effects
        _defect_append_coulomb_deltas!(deltas, mode, c, defects, proposal)
    end
    return deltas
end

"""
    _defect_apply_coulomb_density_deltas!(coulomb, deltas)

Apply the local sheet-charge changes for an accepted defect hop. Spin flips
already keep `ρ` current incrementally, so an accepted charge hop only needs to
patch the affected free-charge sheets before the Coulomb solve.
"""
@inline function _defect_apply_coulomb_density_deltas!(
    c::C,
    deltas::NTuple{N,_DefectCoulombSheetDelta{T}},
) where {C<:CoulombHamiltonian,N,T}
    @inbounds for δ in deltas
        c.ρ[δ.coord] += δ.charge
    end
    return c
end

function _defect_apply_coulomb_density_deltas!(
    c::C,
    deltas::Vector{_DefectCoulombSheetDelta{T}},
) where {C<:CoulombHamiltonian,T}
    @inbounds for δ in deltas
        c.ρ[δ.coord] += δ.charge
    end
    return c
end

@inline _defect_coulomb_mode_count(::Tuple{}) = 0
@inline _defect_coulomb_mode_count(effects::Tuple{<:CoulombChargeCoupling,Vararg}) =
    1 + _defect_coulomb_mode_count(Base.tail(effects))
@inline _defect_coulomb_mode_count(effects::Tuple) =
    _defect_coulomb_mode_count(Base.tail(effects))

@inline _defect_first_coulomb_mode(effects::Tuple{<:CoulombChargeCoupling,Vararg}) = first(effects)
@inline _defect_first_coulomb_mode(effects::Tuple) = _defect_first_coulomb_mode(Base.tail(effects))

"""
    _defect_coulomb_delta_energy(coulomb, deltas)

Calculate `δρ ⋅ u + 1/2 δρ ⋅ Gδρ` for a local free-charge proposal using the
current potential and the precomputed Coulomb Green kernel.
"""
function _defect_coulomb_delta_energy(c::C, kernel::Array{T,4}, deltas::Vector{_DefectCoulombSheetDelta{T}}) where {C<:CoulombHamiltonian,T}
    linear = zero(T)
    self = zero(T)
    @inbounds for δ in deltas
        linear += δ.charge * T(c.u[δ.coord])
    end
    @inbounds for δtarget in deltas, δsource in deltas
        self += δtarget.charge * δsource.charge * _defect_free_charge_green(c, kernel, δtarget.coord, δsource.coord)
    end
    return linear + T(0.5) * self
end

@inline function _defect_coulomb_delta_energy(
    c::C,
    kernel::Array{T,4},
    deltas::NTuple{N,_DefectCoulombSheetDelta{T}},
) where {C<:CoulombHamiltonian,T,N}
    linear = zero(T)
    self = zero(T)
    @inbounds for δ in deltas
        linear += δ.charge * T(c.u[δ.coord])
    end
    @inbounds for δtarget in deltas, δsource in deltas
        self += δtarget.charge * δsource.charge * _defect_free_charge_green(c, kernel, δtarget.coord, δsource.coord)
    end
    return linear + T(0.5) * self
end

"""
    calculate(ΔH(), coulomb, defects, proposal::DefectHopProposal)

Calculate the Coulomb energy change for a trial charged-defect hop from the
cached potential plus the proposal's exact Green self term. This does not solve
Poisson for rejected proposals.
"""
function calculate(::ΔH, c::C, defects::D, proposal::P) where {C<:CoulombHamiltonian,D<:DefectHopping,P<:DefectHopProposal}
    T = eltype(c.ρ)
    proposal.valid || return T(Inf)
    layeridx(c) == Int(defects.layer) || return zero(T)
    _defect_has_coulomb_charge(defects.effects) || return zero(T)

    mode = _defect_first_coulomb_mode(defects.effects)
    kernel = _defect_mode_kernel(mode, c)
    if _defect_coulomb_mode_count(defects.effects) == 1
        deltas = _defect_coulomb_mode_deltas(mode, c, defects, proposal, T)
        return _defect_coulomb_delta_energy(c, kernel, deltas)
    end

    deltas = _defect_coulomb_deltas(c, defects, proposal, T)
    return _defect_coulomb_delta_energy(c, kernel, deltas)
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

    T = eltype(c.ρ)
    mode = _defect_first_coulomb_mode(defects.effects)
    if _defect_coulomb_mode_count(defects.effects) == 1
        deltas = _defect_coulomb_mode_deltas(mode, c, defects, proposal, T)
        _defect_apply_coulomb_effects!(c, defects, proposal.from_idx, -1)
        _defect_apply_coulomb_effects!(c, defects, proposal.to_idx, 1)
        _defect_apply_coulomb_density_deltas!(c, deltas)
        recalc!(c)
        c.recalc_tracker[] = 1
        return nothing
    end

    deltas = _defect_coulomb_deltas(c, defects, proposal, T)
    _defect_apply_coulomb_effects!(c, defects, proposal.from_idx, -1)
    _defect_apply_coulomb_effects!(c, defects, proposal.to_idx, 1)
    _defect_apply_coulomb_density_deltas!(c, deltas)
    recalc!(c)
    c.recalc_tracker[] = 1
    return nothing
end
