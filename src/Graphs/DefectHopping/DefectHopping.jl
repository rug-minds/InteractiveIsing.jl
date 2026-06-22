export DefectHopping, ChargeHopProposer, LocalPotentialShift, ExtFieldShift, ExtFieldChargeCoupling, CoulombChargeShift

abstract type AbstractDefectMode end

"""
    LocalPotentialShift(order, strength; hopping_scale=1)

Mobile-defect mode that shifts mutable `PolynomialHamiltonian{order}` local
potential storage by `strength` at every occupied defect site. `hopping_scale`
only multiplies the Metropolis hop energy contribution, so the visible local
parameter perturbation and the defect mobility energy scale can be tuned
independently.
"""
struct LocalPotentialShift{Order,A,H} <: AbstractDefectMode
    strength::A
    hopping_scale::H
end

LocalPotentialShift(order::I, strength::A = 1; hopping_scale::H = 1) where {I<:Integer,A,H} =
    LocalPotentialShift{Int(order),A,H}(strength, hopping_scale)

LocalPotentialShift{Order}(strength::A = 1; hopping_scale::H = 1) where {Order,A,H} =
    LocalPotentialShift{Order,A,H}(strength, hopping_scale)

"""
    ExtFieldShift(strength; hopping_scale=1)

Mobile-defect mode that shifts mutable `ExtField.b` storage by `strength` at
every occupied defect site. Since `ExtField` contributes `-c*b[i]*s[i]`, a
positive shift favors positive spin when `c > 0`. `hopping_scale` only
multiplies the Metropolis hop energy contribution.
"""
struct ExtFieldShift{A,H} <: AbstractDefectMode
    strength::A
    hopping_scale::H
end

ExtFieldShift(strength::A; hopping_scale::H = 1) where {A,H} =
    ExtFieldShift{A,H}(strength, hopping_scale)

"""
    ExtFieldChargeCoupling(; axis=nothing, hopping_scale=1)

Defect mode that couples a charged hop to the graph's `ExtField` Hamiltonian
term. The external-field value is read from `ExtField.b`, so spin and charged
defect drift share one field parameter. `axis=nothing` uses the last proposal
axis, which is the polarization axis for the current 3D examples.
"""
struct ExtFieldChargeCoupling{A,H} <: AbstractDefectMode
    axis::A
    hopping_scale::H
end

function ExtFieldChargeCoupling(; axis::A = nothing, hopping_scale::H = 1) where {A,H}
    if !(axis isa Nothing)
        axis > 0 || throw(ArgumentError("ExtFieldChargeCoupling axis must be positive; got $axis."))
    end
    return ExtFieldChargeCoupling{A,H}(axis, hopping_scale)
end

"""
    DefectHopping(; layer=1, defects, charge=1)

Proposer for mobile defects represented as polynomial local-potential charges.
Attach this to an `IsingGraph` and run the ordinary `Metropolis()` algorithm to
attempt one nearest-neighbor defect hop per Metropolis step.
"""
struct DefectHopping{L,D,C,E,DI,O,S} <: AbstractProposer
    layer::L
    defects::D
    charge::C
    effects::E
    defect_idxs::DI
    occupancy::O
    state::S
end

"""
    ChargeHopProposer

Combined mobile-charge proposer with separate positive and negative fields.
Metropolis sees one model/proposer, while the two fields keep distinct effects
and occupancies.
"""
struct ChargeHopState{PI,NI,PO,NO}
    positive_idxs::PI
    negative_idxs::NI
    positive_occupancy::PO
    negative_occupancy::NO
end

struct ChargeHopProposer{P,N,CS,G,PR,NR} <: AbstractProposer
    positive::P
    negative::N
    charge_state::CS
    graph::G
    positive_attempt_rate::PR
    negative_attempt_rate::NR
end

"""
    _defect_effect_tuple(effects)

Normalize one user-supplied defect mode or a tuple of modes into a tuple.
"""
@inline _defect_effect_tuple(effects::Tuple) = effects
@inline _defect_effect_tuple(effect::AbstractDefectMode) = (effect,)

"""
    _defect_default_effects(charge)

Return the legacy defect mode for `charge`: a quadratic local-potential shift.
"""
@inline _defect_default_effects(charge::C) where {C} = (LocalPotentialShift(2, charge),)

function DefectHopping(; layer::L = 1, defects::D, charge::C = 1, effects = nothing) where {L,D,C}
    mode_tuple = isnothing(effects) ? _defect_default_effects(charge) : _defect_effect_tuple(effects)
    return DefectHopping{L,D,C,typeof(mode_tuple),Nothing,Nothing,Nothing}(layer, defects, charge, mode_tuple, nothing, nothing, nothing)
end

function DefectHopping(g::G; layer::L = 1, defects::D, charge::C = 1, effects = nothing) where {G<:AbstractIsingGraph,L,D,C}
    mode_tuple = isnothing(effects) ? _defect_default_effects(charge) : _defect_effect_tuple(effects)
    return _defect_bound_system(g, layer, defects, charge, mode_tuple)
end

"""
    ChargeHopProposer(g; positive, negative, positive_effects, negative_effects, layer=1, positive_attempt_rate=1, negative_attempt_rate=1)

Bind one mobile-charge hopping model to `g`. Positive charges typically
represent charged vacancies and may carry structural modes; negative charges
typically represent electron-like carriers and default to Coulomb-only effects.
Attempt rates are per-particle proposal rates; Metropolis selects the positive
or negative field with probability proportional to `rate * count`.
"""
function ChargeHopProposer(
    g::G;
    layer::L = 1,
    positive::P,
    negative::N,
    positive_effects,
    negative_effects,
    positive_charge::PC = 1,
    negative_charge::NC = -1,
    positive_attempt_rate::PAR = 1,
    negative_attempt_rate::NAR = 1,
) where {G<:AbstractIsingGraph,L,P,N,PC,NC,PAR,NAR}
    positive_system = DefectHopping(g; layer, defects = positive, charge = positive_charge, effects = positive_effects)
    negative_system = DefectHopping(g; layer, defects = negative, charge = negative_charge, effects = negative_effects)
    positive_rate = _defect_internal_value(
        positive_attempt_rate,
        physicalunits(role = :dimensionless),
        g,
        :positive_attempt_rate,
    )
    negative_rate = _defect_internal_value(
        negative_attempt_rate,
        physicalunits(role = :dimensionless),
        g,
        :negative_attempt_rate,
    )
    (positive_rate >= zero(positive_rate) && negative_rate >= zero(negative_rate)) ||
        throw(ArgumentError("ChargeHopProposer attempt rates must be nonnegative."))
    (positive_rate > zero(positive_rate) || negative_rate > zero(negative_rate)) ||
        throw(ArgumentError("ChargeHopProposer requires at least one nonzero attempt rate."))
    charge_state = ChargeHopState(
        positive_system.defect_idxs,
        negative_system.defect_idxs,
        positive_system.occupancy,
        negative_system.occupancy,
    )
    return ChargeHopProposer(positive_system, negative_system, charge_state, g, positive_rate, negative_rate)
end

"""
    DefectHopProposal

Internal proposal describing one attempted local-potential defect hop.
"""
struct DefectHopProposal{C,D,E} <: AbstractProposal
    defect_slot::Int
    from_idx::Int
    to_idx::Int
    charge::C
    displacement::D
    effects::E
    valid::Bool
    accepted::Bool
end

"""
    ChargeHopProposal

Wrapper proposal carrying the selected charge field and the underlying
single-field defect-hop proposal.
"""
struct ChargeHopProposal{S,P} <: AbstractProposal
    species::S
    proposal::P
end

function DefectHopProposal(defect_slot::I, from_idx::J, to_idx::K, charge::C, displacement::D, valid::Bool, accepted::Bool) where {I<:Integer,J<:Integer,K<:Integer,C,D}
    effects = _defect_default_effects(charge)
    return DefectHopProposal(Int(defect_slot), Int(from_idx), Int(to_idx), charge, displacement, effects, valid, accepted)
end

"""
    isaccepted(proposal::DefectHopProposal)

Return whether a defect-hop proposal was accepted by Metropolis.
"""
@inline isaccepted(proposal::DefectHopProposal) = proposal.accepted
@inline isaccepted(proposal::ChargeHopProposal) = isaccepted(proposal.proposal)

"""
    _accepted_defect_hop(proposal)

Return an accepted copy of a defect-hop proposal.
"""
@inline function _accepted_defect_hop(proposal::P) where {P<:DefectHopProposal}
    return DefectHopProposal(
        proposal.defect_slot,
        proposal.from_idx,
        proposal.to_idx,
        proposal.charge,
        proposal.displacement,
        proposal.effects,
        proposal.valid,
        true,
    )
end

"""
    _defect_local_index(layer, defect)

Convert a user defect position into a layer-local linear index.
"""
function _defect_local_index(layer::L, defect::I) where {L<:AbstractIsingLayer,I<:Integer}
    local_idx = Int(defect)
    1 <= local_idx <= nStates(layer) ||
        throw(ArgumentError("Defect local index $local_idx is outside layer $(internal_idx(layer)) with length $(nStates(layer))."))
    return local_idx
end

function _defect_local_index(layer::L, defect::CartesianIndex{D}) where {L<:AbstractIsingLayer,D}
    D == length(size(layer)) ||
        throw(ArgumentError("Defect coordinate dimension $D does not match layer dimension $(length(size(layer)))."))
    for axis in 1:D
        1 <= defect[axis] <= size(layer, axis) ||
            throw(ArgumentError("Defect coordinate $defect is outside layer $(internal_idx(layer)) with size $(size(layer))."))
    end
    return Int(LinearIndices(size(layer))[defect])
end

function _defect_local_index(layer::L, defect::NTuple{D,I}) where {L<:AbstractIsingLayer,D,I<:Integer}
    return _defect_local_index(layer, CartesianIndex(defect))
end

"""
    _defect_graph_index(layer, local_idx)

Map a layer-local linear index to the corresponding graph index.
"""
@inline function _defect_graph_index(layer::L, local_idx::I) where {L<:AbstractIsingLayer,I<:Integer}
    return Int(startidx(layer)) + Int(local_idx) - 1
end

"""
    _defect_axis_displacement(rng, top)

Draw one signed coordinate-axis displacement for `top`.
"""
@inline function _defect_axis_displacement(rng::R, top::T) where {R<:AbstractRNG,T<:AbstractLayerTopology}
    axis = rand(rng, 1:ndims(top))
    step = rand(rng, Bool) ? 1 : -1
    return ntuple(i -> i == axis ? step : 0, Val(ndims(top)))
end

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

"""
    _defect_initial_graph_indices(layer, defects)

Validate initial defects and return their graph indices.
"""
function _defect_initial_graph_indices(layer::L, defects::D) where {L<:AbstractIsingLayer,D}
    defect_idxs = Int[]
    seen = Set{Int}()
    for defect in defects
        local_idx = _defect_local_index(layer, defect)
        graph_idx = _defect_graph_index(layer, local_idx)
        graph_idx in seen &&
            throw(ArgumentError("DefectHopping received duplicate defect position at graph index $graph_idx."))
        push!(seen, graph_idx)
        push!(defect_idxs, graph_idx)
    end
    isempty(defect_idxs) && throw(ArgumentError("DefectHopping requires at least one initial defect."))
    return defect_idxs
end

"""
    _defect_bound_system(g, layer, defects, charge, effects)

Create a bound defect-hopping system for `g` and apply the configured initial
defect effects.
"""
function _defect_bound_system(g::G, layer_arg::L, defects::D, charge_arg::C, effects_arg::E) where {G<:AbstractIsingGraph,L,D,C,E<:Tuple}
    layer_idx = Int(layer_arg)
    layer = g[layer_idx]
    effects = _defect_convert_effects(effects_arg, g)
    charge = _defect_convert_charge(charge_arg, effects, g)
    defect_idxs = _defect_initial_graph_indices(layer, defects)

    _defect_validate_effects!(g.hamiltonian, effects, first(defect_idxs))
    effects = _defect_bind_effects(g.hamiltonian, effects, g, layer_arg)

    occupancy = falses(nstates(g))
    for idx in defect_idxs
        @inbounds occupancy[idx] = true
        _defect_apply_initial_effects!(g.hamiltonian, effects, g, layer_arg, idx, 1)
    end

    return DefectHopping(layer_arg, defects, charge, effects, defect_idxs, occupancy, g)
end

"""
    bind_proposer(g, proposer::DefectHopping)

Bind defect hopping to `g`, allocate occupancy, and apply the initial polynomial
local-potential charge once.
"""
function bind_proposer(g::G, proposer::P) where {G<:AbstractIsingGraph,P<:DefectHopping}
    !isnothing(proposer.state) && !isnothing(proposer.defect_idxs) && return proposer

    bound = _defect_bound_system(g, proposer.layer, proposer.defects, proposer.charge, proposer.effects)

    # Store the bound proposer back on the mutable graph so repeated Metropolis
    # initialization keeps the current defect positions and does not reapply the
    # initial charge.
    setproperty!(g, :proposer, bound)
    return bound
end

"""
    get_proposer(defects::DefectHopping)

Return a bound defect-hopping system as the proposer for Metropolis.
"""
function get_proposer(defects::D) where {D<:DefectHopping}
    isnothing(defects.state) &&
        throw(ArgumentError("Construct explicit defect systems as `DefectHopping(g; defects = ...)` before passing them to Metropolis."))
    isnothing(defects.defect_idxs) &&
        throw(ArgumentError("DefectHopping model is not bound. Use `DefectHopping(g; defects = ...)`."))
    return defects
end

"""
    get_proposer(charges::ChargeHopProposer)

Return the combined positive/negative charge model as its own proposer.
"""
function get_proposer(charges::C) where {C<:ChargeHopProposer}
    return charges
end

"""
    graphstate(defects::DefectHopping)

Return the spin state used by the graph that owns this defect system.
"""
@inline graphstate(defects::D) where {D<:DefectHopping} = graphstate(defects.state)

@inline Base.eltype(defects::D) where {D<:DefectHopping} = _defect_parameter_type(defects.state)
@inline temp(defects::D) where {D<:DefectHopping} = temp(defects.state)

"""
    state(charges::ChargeHopProposer)

Return the grouped positive/negative charge state for a mobile-charge hopping
model.
"""
@inline state(charges::C) where {C<:ChargeHopProposer} = charges.charge_state

"""
    graphstate(charges::ChargeHopProposer)

Return the spin state of the graph coupled to a mobile-charge hopping model.
"""
@inline graphstate(charges::C) where {C<:ChargeHopProposer} = graphstate(charges.graph)

@inline Base.eltype(charges::C) where {C<:ChargeHopProposer} = _defect_parameter_type(charges.graph)
@inline temp(charges::C) where {C<:ChargeHopProposer} = temp(charges.graph)

function Base.getproperty(defects::D, name::Symbol) where {D<:DefectHopping}
    name === :hamiltonian && return getproperty(getfield(defects, :state), :hamiltonian)
    return getfield(defects, name)
end

function Base.getproperty(charges::C, name::Symbol) where {C<:ChargeHopProposer}
    name === :hamiltonian && return getproperty(getfield(charges, :graph), :hamiltonian)
    name === :state && return getfield(charges, :charge_state)
    return getfield(charges, name)
end

"""
    _defect_reapply_initialized_effects!(hamiltonian, defects)

Reapply defect effects that live in Hamiltonian internals rebuilt by `init!`.
Ordinary local-potential modes mutate parameter storage directly and therefore
do not need a reapply step.
"""
function _defect_reapply_initialized_effects!(hamiltonian::H, defects::D) where {H<:Hamiltonian,D<:DefectHopping}
    return hamiltonian
end

function init!(hts::HamiltonianTerms{Hs}, defects::D) where {Hs,D<:DefectHopping}
    initialized = init!(hts, defects.state)
    return _defect_reapply_initialized_effects!(initialized, defects)
end

function init!(hts::H, defects::D) where {H<:Hamiltonian,D<:DefectHopping}
    initialized = init!(hts, defects.state)
    return _defect_reapply_initialized_effects!(initialized, defects)
end

function init!(hts::HamiltonianTerms{Hs}, charges::C) where {Hs,C<:ChargeHopProposer}
    return init!(hts, charges.graph)
end

function init!(hts::H, charges::C) where {H<:Hamiltonian,C<:ChargeHopProposer}
    return init!(hts, charges.graph)
end

"""
    rand(rng, proposer::DefectHopping)

Draw one axis-neighbor hop proposal from the current defect occupancy.
"""
function Base.rand(rng::R, proposer::P) where {R<:AbstractRNG,P<:DefectHopping}
    isnothing(proposer.state) &&
        throw(ArgumentError("DefectHopping must be attached to an IsingGraph before drawing proposals."))

    model = proposer.state
    layer = model[Int(proposer.layer)]
    top = topology(layer)
    linear = LinearIndices(size(layer))

    defect_slot = rand(rng, 1:length(proposer.defect_idxs))
    from_idx = @inbounds proposer.defect_idxs[defect_slot]
    local_from = Int(from_idx - startidx(layer) + 1)
    from_coord = CartesianIndices(size(layer))[local_from]
    displacement = _defect_axis_displacement(rng, top)

    # Work directly with Cartesian indices here because `Coordinate(top, i)` is
    # ambiguous for one-dimensional topologies.
    top_size = size(top)
    periodic_axes = whichperiodic(top)
    to_coord = ntuple(Val(ndims(top))) do axis
        raw = from_coord[axis] + displacement[axis]
        periodic_axes[axis] ? mod1(raw, top_size[axis]) : raw
    end

    # Non-periodic boundary crossings and occupied target sites are marked
    # invalid so Metropolis rejects the move.
    valid = all(ntuple(axis -> 1 <= to_coord[axis] <= top_size[axis], Val(ndims(top))))
    to_idx = from_idx
    if valid
        local_to = Int(linear[CartesianIndex(to_coord)])
        to_idx = _defect_graph_index(layer, local_to)
        valid = !(@inbounds proposer.occupancy[to_idx])
    end

    return DefectHopProposal(defect_slot, from_idx, to_idx, proposer.charge, displacement, proposer.effects, valid, false)
end

"""
    rand(proposer::DefectHopping)

Draw a defect-hop proposal with Julia's default RNG.
"""
@inline Base.rand(proposer::DefectHopping) = rand(Random.default_rng(), proposer)

"""
    rand(rng, proposer::ChargeHopProposer)

Draw one hop from either the positive or negative charge field.
"""
function Base.rand(rng::R, proposer::P) where {R<:AbstractRNG,P<:ChargeHopProposer}
    npositive = length(proposer.positive.defect_idxs)
    nnegative = length(proposer.negative.defect_idxs)
    positive_weight = proposer.positive_attempt_rate * npositive
    negative_weight = proposer.negative_attempt_rate * nnegative
    total_weight = positive_weight + negative_weight
    total_weight > zero(total_weight) ||
        throw(ArgumentError("ChargeHopProposer has no active proposal species; check attempt rates and occupancies."))

    if rand(rng) * total_weight < positive_weight
        return ChargeHopProposal(PositiveFreeCharge(), rand(rng, proposer.positive))
    else
        return ChargeHopProposal(NegativeFreeCharge(), rand(rng, proposer.negative))
    end
end

@inline Base.rand(proposer::ChargeHopProposer) = rand(Random.default_rng(), proposer)

"""
    accept(proposer, proposal::DefectHopProposal)

Commit an accepted defect hop to proposer occupancy and return an accepted
proposal. Hamiltonian local-potential storage is updated by `update!`.
"""
function accept(proposer::P, proposal::DP) where {P<:DefectHopping,DP<:DefectHopProposal}
    proposal.valid || return proposal

    @inbounds begin
        proposer.occupancy[proposal.from_idx] = false
        proposer.occupancy[proposal.to_idx] = true
        proposer.defect_idxs[proposal.defect_slot] = proposal.to_idx
    end

    return _accepted_defect_hop(proposal)
end

"""
    accept(proposer::ChargeHopProposer, proposal::ChargeHopProposal)

Commit the selected positive or negative charge hop to the corresponding field.
"""
function accept(proposer::P, proposal::NP) where {P<:ChargeHopProposer,NP<:ChargeHopProposal{PositiveFreeCharge}}
    return ChargeHopProposal(proposal.species, accept(proposer.positive, proposal.proposal))
end

function accept(proposer::P, proposal::NP) where {P<:ChargeHopProposer,NP<:ChargeHopProposal{NegativeFreeCharge}}
    return ChargeHopProposal(proposal.species, accept(proposer.negative, proposal.proposal))
end

@inline _charge_hop_system(model::M, proposal::P) where {M<:ChargeHopProposer,P<:ChargeHopProposal{PositiveFreeCharge}} =
    model.positive

@inline _charge_hop_system(model::M, proposal::P) where {M<:ChargeHopProposer,P<:ChargeHopProposal{NegativeFreeCharge}} =
    model.negative

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
    calculate(ΔH(), hamiltonian_terms, charges, proposal::ChargeHopProposal)

Route a combined charge proposal to the selected positive or negative field.
"""
@inline function calculate(dh::ΔH, hts::HTS, model::M, proposal::P) where {HTS<:AbstractHamiltonianTerms,M<:ChargeHopProposer,P<:ChargeHopProposal}
    return calculate(dh, hts, _charge_hop_system(model, proposal), proposal.proposal)
end

@inline function calculate(dh::ΔH, hterm::H, model::M, proposal::P) where {H<:Hamiltonian,M<:ChargeHopProposer,P<:ChargeHopProposal}
    return calculate(dh, hterm, _charge_hop_system(model, proposal), proposal.proposal)
end

@inline function calculate(dh::ΔH, hterm::H, model::M, proposal::P) where {H<:HamiltonianTerm,M<:ChargeHopProposer,P<:ChargeHopProposal}
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
        total += _defect_delta(mode, hterm, T, from_state, to_state)
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

"""
    update!(::Metropolis, hterm, charges, proposal::ChargeHopProposal)

Route accepted combined charge proposals to the selected positive or negative
field's existing update methods.
"""
@inline function update!(algo::A, hterm::H, model::M, proposal::P) where {A<:Metropolis,H<:Hamiltonian,M<:ChargeHopProposer,P<:ChargeHopProposal}
    return update!(algo, hterm, _charge_hop_system(model, proposal), proposal.proposal)
end

@inline function update!(algo::A, hts::HTS, model::M, proposal::P) where {A<:Metropolis,HTS<:AbstractHamiltonianTerms,M<:ChargeHopProposer,P<:ChargeHopProposal}
    return update!(algo, hts, _charge_hop_system(model, proposal), proposal.proposal)
end

@inline function update!(algo::A, hts::HTS, model::M, proposal::P) where {A<:Metropolis,Hs,HTS<:HamiltonianTerms{Hs},M<:ChargeHopProposer,P<:ChargeHopProposal}
    return update!(algo, hts, _charge_hop_system(model, proposal), proposal.proposal)
end

@inline function update!(algo::A, hterm::H, model::M, proposal::P) where {A<:Metropolis,H<:HamiltonianTerm,M<:ChargeHopProposer,P<:ChargeHopProposal}
    return update!(algo, hterm, _charge_hop_system(model, proposal), proposal.proposal)
end

include("CoulombCoupling.jl")
