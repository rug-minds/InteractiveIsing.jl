"""
    LocalPotentialShiftCoupling(order, strength; hopping_scale=1)

Mobile-species coupling hamiltonian that shifts mutable
`PolynomialHamiltonian{order}` local potential storage by `strength` at every
occupied site. `hopping_scale` only multiplies the Metropolis hop energy
contribution, so the visible perturbation and the hopping barrier can be tuned
independently.
"""
struct LocalPotentialShiftCoupling{Order,A,H} <: AbstractDefectCoupling
    strength::A
    hopping_scale::H
end

LocalPotentialShiftCoupling(order::I, strength::A = 1; hopping_scale::H = 1) where {I<:Integer,A,H} =
    LocalPotentialShiftCoupling{Int(order),A,H}(strength, hopping_scale)

LocalPotentialShiftCoupling{Order}(strength::A = 1; hopping_scale::H = 1) where {Order,A,H} =
    LocalPotentialShiftCoupling{Order,A,H}(strength, hopping_scale)

const LocalPotentialShift = LocalPotentialShiftCoupling

"""
    LocalPotentialScaleCoupling(order, factor; hopping_scale=1)

Mobile-species coupling hamiltonian that multiplicatively scales mutable
`PolynomialHamiltonian{order}` local potential storage at occupied sites. An
accepted hop divides the old site by `factor` and multiplies the new site by
`factor`, so this term should be used only with nonzero scale factors.
"""
struct LocalPotentialScaleCoupling{Order,F,H} <: AbstractDefectCoupling
    factor::F
    hopping_scale::H
end

function LocalPotentialScaleCoupling(order::I, factor::F = 1; hopping_scale::H = 1) where {I<:Integer,F,H}
    factor == zero(factor) && throw(ArgumentError("LocalPotentialScaleCoupling factor must be nonzero."))
    return LocalPotentialScaleCoupling{Int(order),F,H}(factor, hopping_scale)
end

function LocalPotentialScaleCoupling{Order}(factor::F = 1; hopping_scale::H = 1) where {Order,F,H}
    factor == zero(factor) && throw(ArgumentError("LocalPotentialScaleCoupling factor must be nonzero."))
    return LocalPotentialScaleCoupling{Order,F,H}(factor, hopping_scale)
end

"""
    ExtFieldShiftCoupling(strength; hopping_scale=1)

Mobile-species coupling hamiltonian that shifts mutable `ExtField.b` storage by
`strength` at every occupied site. Since `ExtField` contributes `-c*b[i]*s[i]`,
a positive shift favors positive spin when `c > 0`. `hopping_scale` only
multiplies the Metropolis hop energy contribution.
"""
struct ExtFieldShiftCoupling{A,H} <: AbstractDefectCoupling
    strength::A
    hopping_scale::H
end

ExtFieldShiftCoupling(strength::A; hopping_scale::H = 1) where {A,H} =
    ExtFieldShiftCoupling{A,H}(strength, hopping_scale)

const ExtFieldShift = ExtFieldShiftCoupling
const ExternalFieldShiftCoupling = ExtFieldShiftCoupling

"""
    ExtFieldChargeCoupling(; axis=nothing, hopping_scale=1)

Defect mode that couples a charged hop to the graph's `ExtField` Hamiltonian
term. The external-field value is read from `ExtField.b`, so spin and charged
defect drift share one field parameter. `axis=nothing` uses the last proposal
axis, which is the polarization axis for the current 3D examples.
"""
struct ExtFieldChargeCoupling{A,H} <: AbstractDefectCoupling
    axis::A
    hopping_scale::H
end

function ExtFieldChargeCoupling(; axis::A = nothing, hopping_scale::H = 1) where {A,H}
    if !(axis isa Nothing)
        axis > 0 || throw(ArgumentError("ExtFieldChargeCoupling axis must be positive; got $axis."))
    end
    return ExtFieldChargeCoupling{A,H}(axis, hopping_scale)
end

const ExternalFieldChargeCoupling = ExtFieldChargeCoupling

"""
    _defect_effect_tuple(effects)

Normalize one user-supplied coupling hamiltonian, `HamiltonianTerms` chain, or
legacy tuple of couplings into the internal tuple representation.
"""
@inline _defect_effect_tuple(effects::Tuple) = effects
@inline _defect_effect_tuple(effect::AbstractDefectCoupling) = (effect,)
@inline _defect_effect_tuple(effects::HamiltonianTerms) = hamiltonians(effects)

"""
    _defect_default_effects(charge)

Return the legacy defect coupling for `charge`: a quadratic local-potential
shift.
"""
@inline _defect_default_effects(charge::C) where {C} = (LocalPotentialShift(2, charge),)

"""
    _defect_lp_error(lp)

Build the local-potential storage error used when defect charges cannot mutate
independent sites.
"""
function _defect_lp_error(lp)
    return ArgumentError(
        "DefectHopping requires mutable site-wise local parameter storage; got $(typeof(lp)). " *
        "Pass localpotential = Vector, OffsetArray, or another independently mutable vector-like storage.",
    )
end

"""
    _defect_uses_localpotential(lp)

Return whether a polynomial local-potential array should carry mobile defect
charge. Constant or uniform polynomial terms are treated as background terms.
"""
@inline function _defect_uses_localpotential(lp)
    return lp isa AbstractVector && !(lp isa ConstFill) && !(lp isa UniformArray)
end

"""
    _defect_internal_value(value, units, graph, parameter)

Convert a defect-hopping scalar with the same physical scale context used by
Hamiltonian parameters, then cast it to the graph precision.
"""
@inline function _defect_internal_value(value::V, units::U, g::G, parameter::Symbol) where {V,U,G<:AbstractIsingGraph}
    T = _defect_parameter_type(g)
    return convert(T, internalvalue(value, units, physicalscales(g), g; parameter))
end

"""
    _defect_parameter_type(graph)

Return the numeric type used for bound defect-hopping scalar parameters.
Defect hopping mutates Hamiltonian parameter storage and evaluates energies
against the graph spin state, so hopping scalars deliberately follow the
element type of `graphstate(graph)`.
"""
@inline function _defect_parameter_type(g::G) where {G<:AbstractIsingGraph}
    return eltype(graphstate(g))
end

@inline function _defect_convert_mode(mode::LocalPotentialShift{Order}, g::G) where {Order,G<:AbstractIsingGraph}
    strength = _defect_internal_value(mode.strength, physicalunits(role = :dimensionless), g, :local_potential_shift)
    hopping_scale = _defect_internal_value(mode.hopping_scale, physicalunits(role = :dimensionless), g, :local_potential_hopping_scale)
    return LocalPotentialShift{Order}(strength; hopping_scale)
end

@inline function _defect_convert_mode(mode::LocalPotentialScaleCoupling{Order}, g::G) where {Order,G<:AbstractIsingGraph}
    factor = _defect_internal_value(mode.factor, physicalunits(role = :dimensionless), g, :local_potential_scale)
    hopping_scale = _defect_internal_value(mode.hopping_scale, physicalunits(role = :dimensionless), g, :local_potential_scale_hopping_scale)
    return LocalPotentialScaleCoupling{Order}(factor; hopping_scale)
end

@inline function _defect_convert_mode(mode::ExtFieldShift, g::G) where {G<:AbstractIsingGraph}
    strength = _defect_internal_value(mode.strength, physicalunits(energy = 1, role = :field_energy), g, :extfield_shift)
    hopping_scale = _defect_internal_value(mode.hopping_scale, physicalunits(role = :dimensionless), g, :extfield_hopping_scale)
    return ExtFieldShift(strength; hopping_scale)
end

@inline function _defect_convert_mode(mode::ExtFieldChargeCoupling, g::G) where {G<:AbstractIsingGraph}
    hopping_scale = _defect_internal_value(mode.hopping_scale, physicalunits(role = :dimensionless), g, :extfield_charge_hopping_scale)
    return ExtFieldChargeCoupling(; axis = mode.axis, hopping_scale)
end

@inline function _defect_convert_effects(effects::Tuple, g::G) where {G<:AbstractIsingGraph}
    return map(effect -> _defect_convert_mode(effect, g), effects)
end

"""
    _defect_charge_uses_physical_units(effects)

Return whether the proposal charge should be converted as physical free charge
instead of a dimensionless legacy local-potential amplitude.
"""
@inline _defect_mode_uses_physical_charge(::AbstractDefectMode) = false
@inline _defect_mode_uses_physical_charge(::ExtFieldChargeCoupling) = true
@inline _defect_charge_uses_physical_units(::Tuple{}) = false
@inline function _defect_charge_uses_physical_units(effects::Tuple)
    return _defect_mode_uses_physical_charge(first(effects)) || _defect_charge_uses_physical_units(Base.tail(effects))
end

"""
    _defect_convert_charge(charge, effects, graph)

Convert the proposal charge to graph precision, using charge units only for
effects that model mobile free charge.
"""
@inline function _defect_convert_charge(charge::C, effects::E, g::G) where {C,E<:Tuple,G<:AbstractIsingGraph}
    units = _defect_charge_uses_physical_units(effects) ?
        physicalunits(charge = 1, role = :free_charge) :
        physicalunits(role = :dimensionless)
    return _defect_internal_value(charge, units, g, :defect_charge)
end

"""
    _defect_validate_mode!(mode, hterm, idx)

Return true when `hterm` is targeted by `mode` and has mutable site-wise storage.
"""
function _defect_validate_mode!(mode::LocalPotentialShift{Order}, hterm::H, idx::I) where {Order,H<:PolynomialHamiltonian{Order},I<:Integer}
    lp = hterm.lp
    _defect_uses_localpotential(lp) || return false
    try
        old = @inbounds lp[Int(idx)]
        @inbounds lp[Int(idx)] = old
    catch
        throw(_defect_lp_error(lp))
    end
    return true
end

function _defect_validate_mode!(mode::LocalPotentialScaleCoupling{Order}, hterm::H, idx::I) where {Order,H<:PolynomialHamiltonian{Order},I<:Integer}
    lp = hterm.lp
    _defect_uses_localpotential(lp) || return false
    try
        old = @inbounds lp[Int(idx)]
        @inbounds lp[Int(idx)] = old
    catch
        throw(_defect_lp_error(lp))
    end
    mode.factor == zero(mode.factor) &&
        throw(ArgumentError("LocalPotentialScaleCoupling factor must be nonzero."))
    return true
end

function _defect_validate_mode!(mode::ExtFieldShift, hterm::H, idx::I) where {H<:ExtField,I<:Integer}
    b = hterm.b
    _defect_uses_localpotential(b) || return false
    try
        old = @inbounds b[Int(idx)]
        @inbounds b[Int(idx)] = old
    catch
        throw(_defect_lp_error(b))
    end
    return true
end

function _defect_validate_mode!(mode::ExtFieldChargeCoupling, hterm::H, idx::I) where {H<:ExtField,I<:Integer}
    return true
end

function _defect_validate_mode!(mode::AbstractDefectMode, hterm::H, idx::I) where {H<:Hamiltonian,I<:Integer}
    return false
end

"""
    _defect_validate_mode!(mode, hamiltonian, idx)

Validate one mode against a Hamiltonian term collection.
"""
function _defect_validate_mode!(mode::M, hts::HTS, idx::I) where {M<:AbstractDefectMode,HTS<:AbstractHamiltonianTerms,I<:Integer}
    found = false
    for hterm in hamiltonians(hts)
        found |= _defect_validate_mode!(mode, hterm, idx)
    end
    return found
end

"""
    _defect_validate_effects!(hamiltonian, effects, idx)

Validate that every defect mode has at least one mutable matching Hamiltonian
parameter to mutate.
"""
function _defect_validate_effects!(hamiltonian::H, effects::E, idx::I) where {H<:Hamiltonian,E<:Tuple,I<:Integer}
    isempty(effects) && throw(ArgumentError("DefectHopping requires at least one defect effect mode."))
    for mode in effects
        _defect_validate_mode!(mode, hamiltonian, idx) ||
            throw(ArgumentError("DefectHopping mode $(typeof(mode)) did not find mutable matching Hamiltonian storage."))
    end
    return nothing
end

"""
    _defect_bind_effects(hamiltonian, effects, graph, layer)

Bind effect modes to Hamiltonian-local caches needed by specialized defect
couplings. Ordinary local modes are returned unchanged.
"""
@inline function _defect_bind_mode(mode::M, hterm::H, g::G, layer_arg::L) where {M<:AbstractDefectMode,H<:Hamiltonian,G<:AbstractIsingGraph,L}
    return mode
end

function _defect_bind_mode(mode::M, hts::HTS, g::G, layer_arg::L) where {M<:AbstractDefectMode,HTS<:AbstractHamiltonianTerms,G<:AbstractIsingGraph,L}
    for hterm in hamiltonians(hts)
        bound = _defect_bind_mode(mode, hterm, g, layer_arg)
        typeof(bound) === typeof(mode) || return bound
    end
    return mode
end

@inline function _defect_bind_effects(hamiltonian::H, effects::E, g::G, layer_arg::L) where {H<:Hamiltonian,E<:Tuple,G<:AbstractIsingGraph,L}
    return map(mode -> _defect_bind_mode(mode, hamiltonian, g, layer_arg), effects)
end

"""
    _defect_apply_mode!(mode, hamiltonian, idx, sign)

Apply one mode's local parameter shift at `idx`, with `sign` equal to `+1` or
`-1`.
"""
function _defect_apply_mode!(mode::M, hts::HTS, idx::I, sign::S) where {M<:AbstractDefectMode,HTS<:AbstractHamiltonianTerms,I<:Integer,S}
    for hterm in hamiltonians(hts)
        _defect_apply_mode!(mode, hterm, idx, sign)
    end
    return nothing
end

function _defect_apply_mode!(mode::LocalPotentialShift{Order}, hterm::H, idx::I, sign::S) where {Order,H<:PolynomialHamiltonian{Order},I<:Integer,S}
    _defect_uses_localpotential(hterm.lp) || return nothing
    @inbounds hterm.lp[Int(idx)] += sign * mode.strength
    return nothing
end

function _defect_apply_mode!(mode::LocalPotentialScaleCoupling{Order}, hterm::H, idx::I, sign::S) where {Order,H<:PolynomialHamiltonian{Order},I<:Integer,S}
    _defect_uses_localpotential(hterm.lp) || return nothing
    @inbounds begin
        if sign > 0
            hterm.lp[Int(idx)] *= mode.factor
        else
            hterm.lp[Int(idx)] /= mode.factor
        end
    end
    return nothing
end

function _defect_apply_mode!(mode::ExtFieldShift, hterm::H, idx::I, sign::S) where {H<:ExtField,I<:Integer,S}
    _defect_uses_localpotential(hterm.b) || return nothing
    @inbounds hterm.b[Int(idx)] += sign * mode.strength
    return nothing
end

function _defect_apply_mode!(mode::ExtFieldChargeCoupling, hterm::H, idx::I, sign::S) where {H<:Hamiltonian,I<:Integer,S}
    return nothing
end

function _defect_apply_mode!(mode::AbstractDefectMode, hterm::H, idx::I, sign::S) where {H<:Hamiltonian,I<:Integer,S}
    return nothing
end

"""
    _defect_apply_effects!(hamiltonian, effects, idx, sign)

Apply every configured defect mode at one graph index.
"""
function _defect_apply_effects!(hamiltonian::H, effects::E, idx::I, sign::S) where {H<:Hamiltonian,E<:Tuple,I<:Integer,S}
    for mode in effects
        _defect_apply_mode!(mode, hamiltonian, idx, sign)
    end
    return nothing
end

"""
    _defect_apply_initial_effects!(hamiltonian, effects, graph, layer, idx, sign)

Apply defect effects during binding, when graph/layer geometry is still
available for modes that target nonlocal Hamiltonian internals.
"""
function _defect_apply_initial_mode!(mode::M, hts::HTS, g::G, layer_arg::L, idx::I, sign::S) where {M<:AbstractDefectMode,HTS<:AbstractHamiltonianTerms,G<:AbstractIsingGraph,L,I<:Integer,S}
    for hterm in hamiltonians(hts)
        _defect_apply_initial_mode!(mode, hterm, g, layer_arg, idx, sign)
    end
    return nothing
end

function _defect_apply_initial_mode!(mode::M, hterm::H, g::G, layer_arg::L, idx::I, sign::S) where {M<:AbstractDefectMode,H<:Hamiltonian,G<:AbstractIsingGraph,L,I<:Integer,S}
    _defect_apply_mode!(mode, hterm, idx, sign)
    return nothing
end

function _defect_apply_initial_effects!(hamiltonian::H, effects::E, g::G, layer_arg::L, idx::I, sign::S) where {H<:Hamiltonian,E<:Tuple,G<:AbstractIsingGraph,L,I<:Integer,S}
    for mode in effects
        _defect_apply_initial_mode!(mode, hamiltonian, g, layer_arg, idx, sign)
    end
    return nothing
end
