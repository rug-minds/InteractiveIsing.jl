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
    MobileChargeState

Grouped state view for a mobile charge-hopping model. Vacancies and charges keep
separate index vectors and occupancy masks, matching the two-species conserved
state stored in the Coulomb term.
"""
struct MobileChargeState{VI,CI,VO,CO}
    vacancy_idxs::VI
    charge_idxs::CI
    vacancy_occupancy::VO
    charge_occupancy::CO
end

const ChargeHopState = MobileChargeState

"""
    MobileChargeSpecies(initial; charge, hamiltonian)

Species construction payload used by `MobileVacancies` and `MobileCharges`.
`initial` may be an integer count for random initialization or an explicit
collection of positions.
"""
struct MobileChargeSpecies{Name,I,C,H}
    initial::I
    charge::C
    hamiltonian::H
end

"""
    _defect_species_hamiltonian(species)

Return the normalized coupling-hamiltonian tuple stored on a mobile species
entry.
"""
@inline function _defect_species_hamiltonian(species::S) where {S<:MobileChargeSpecies}
    return _defect_effect_tuple(species.hamiltonian)
end

"""
    MobileVacancies(initial; charge=1, hamiltonian=nothing)

Create the vacancy entry for `DefectsModel`. Pass an integer `initial`
for random placement, or a collection of coordinates/indices for manual
initialization.
"""
function MobileVacancies(initial::I; charge::C = 1, hamiltonian = nothing) where {I,C}
    h = isnothing(hamiltonian) ? CoulombChargeCoupling(charge) : hamiltonian
    return MobileChargeSpecies{:vacancies,I,C,typeof(h)}(initial, charge, h)
end

"""
    MobileCharges(initial; charge=-1, hamiltonian=nothing)

Create the mobile charge/carrier entry for `DefectsModel`. Pass an
integer `initial` for random placement, or a collection of coordinates/indices
for manual initialization.
"""
function MobileCharges(initial::I; charge::C = -1, hamiltonian = nothing) where {I,C}
    h = isnothing(hamiltonian) ? CoulombChargeCoupling(charge) : hamiltonian
    return MobileChargeSpecies{:charges,I,C,typeof(h)}(initial, charge, h)
end

"""
    DefectsModel

Add-on Monte Carlo model for two mobile defect species coupled to an
`IsingGraph`. It owns the vacancy and carrier state and can be stepped by
`Metropolis`, while `get_proposer(model)` derives a `ChargeHopProposer` that
draws `ChargeHopProposal`s.
"""
struct DefectsModel{T,V,C,CS,G,VR,CR} <: AddOnAbstractMonteCarloModel{T}
    vacancies::V
    charges::C
    charge_state::CS
    graph::G
    vacancy_attempt_rate::VR
    charge_attempt_rate::CR
end

"""
    DefectsModel(vacancies, charges, charge_state, graph, vacancy_rate, charge_rate)

Construct a bound defects model from already-bound species systems. This keeps
the numeric type in the model type parameters, following the wrapped graph.
"""
function DefectsModel(vacancies::V, charges::C, charge_state::CS, graph::G, vacancy_rate::VR, charge_rate::CR) where {V,C,CS,G<:AbstractIsingGraph,VR,CR}
    T = _defect_parameter_type(graph)
    return DefectsModel{T,V,C,CS,G,VR,CR}(vacancies, charges, charge_state, graph, vacancy_rate, charge_rate)
end

const MobileChargeHopping = DefectsModel

Base.eltype(::DefectsModel{T}) where {T} = T
Base.eltype(::Type{<:DefectsModel{T}}) where {T} = T

"""
    requires(::Type{<:DefectsModel})

Declare that defects models require an `AbstractIsingGraph` dependency before
they can be initialized.
"""
requires(::Type{<:DefectsModel}) = (AbstractIsingGraph,)

"""
    dependson(model::DefectsModel)

Return the concrete graph model that this add-on defects model depends on.
"""
dependson(model::M) where {M<:DefectsModel} = model.graph

"""
    ChargeHopProposer(model)

Derived proposal source for `DefectsModel`. The proposer has no separate
state ownership; accepted proposals mutate the species fields held by `model`.
"""
struct ChargeHopProposer{M<:DefectsModel} <: AbstractProposer
    model::M
end

function DefectHopping(; layer::L = 1, defects::D, charge::C = 1, effects = nothing, hamiltonian = nothing) where {L,D,C}
    coupling_source = isnothing(hamiltonian) ? effects : hamiltonian
    mode_tuple = isnothing(coupling_source) ? _defect_default_effects(charge) : _defect_effect_tuple(coupling_source)
    return DefectHopping{L,D,C,typeof(mode_tuple),Nothing,Nothing,Nothing}(layer, defects, charge, mode_tuple, nothing, nothing, nothing)
end

function DefectHopping(g::G; layer::L = 1, defects::D, charge::C = 1, effects = nothing, hamiltonian = nothing) where {G<:AbstractIsingGraph,L,D,C}
    coupling_source = isnothing(hamiltonian) ? effects : hamiltonian
    mode_tuple = isnothing(coupling_source) ? _defect_default_effects(charge) : _defect_effect_tuple(coupling_source)
    return _defect_bound_system(g, layer, defects, charge, mode_tuple)
end

"""
    _mobile_species_from_input(input; charge, hamiltonian, constructor)

Normalize either a fully specified `MobileChargeSpecies` or a count/position
payload plus separate charge/hamiltonian keywords into one species entry.
"""
@inline function _mobile_species_from_input(
    species::S;
    charge,
    hamiltonian,
    constructor,
) where {S<:MobileChargeSpecies}
    return species
end

@inline function _mobile_species_from_input(
    initial;
    charge,
    hamiltonian,
    constructor::F,
) where {F}
    return constructor(initial; charge, hamiltonian)
end

"""
    _random_defect_positions(rng, layer, count, forbidden)

Draw unique random layer coordinates for a mobile species, avoiding any graph
indices in `forbidden`.
"""
function _random_defect_positions(rng::R, layer::L, count::I, forbidden) where {R<:AbstractRNG,L<:AbstractIsingLayer,I<:Integer}
    count >= 0 || throw(ArgumentError("Mobile charge species count must be nonnegative; got $count."))
    available = nStates(layer) - length(forbidden)
    count <= available ||
        throw(ArgumentError("Cannot place $count mobile charges on layer $(internal_idx(layer)); only $available unoccupied sites are available."))

    cartesian = CartesianIndices(size(layer))
    selected = Vector{eltype(cartesian)}()
    seen = Set{Int}(forbidden)
    while length(selected) < count
        local_idx = rand(rng, 1:nStates(layer))
        graph_idx = _defect_graph_index(layer, local_idx)
        graph_idx in seen && continue
        push!(seen, graph_idx)
        push!(selected, cartesian[local_idx])
    end
    return selected
end

"""
    _mobile_species_initial_positions(rng, graph, layer, species, forbidden)

Resolve a species entry to explicit positions. Integer entries request random
initialization; collections are used as manual positions.
"""
function _mobile_species_initial_positions(rng::R, g::G, layer_arg::L, species::S, forbidden) where {R<:AbstractRNG,G<:AbstractIsingGraph,L,S<:MobileChargeSpecies}
    layer = g[Int(layer_arg)]
    initial = species.initial
    initial isa Integer && return _random_defect_positions(rng, layer, initial, forbidden)
    return initial
end

"""
    _mobile_attempt_rates(graph; vacancy_attempt_rate, charge_attempt_rate, electron_attempt_rate)

Resolve relative vacancy/carrier proposal rates. Supplying only one side uses
the other side as the unit rate; supplying neither gives equal rates.
"""
function _mobile_attempt_rates(
    g::G;
    vacancy_attempt_rate = nothing,
    charge_attempt_rate = nothing,
    electron_attempt_rate = nothing,
) where {G<:AbstractIsingGraph}
    !isnothing(charge_attempt_rate) && !isnothing(electron_attempt_rate) &&
        throw(ArgumentError("Pass only one of `charge_attempt_rate` or `electron_attempt_rate`."))
    carrier_rate_arg = isnothing(charge_attempt_rate) ? electron_attempt_rate : charge_attempt_rate
    !isnothing(vacancy_attempt_rate) && !isnothing(carrier_rate_arg) &&
        throw(ArgumentError("Pass either `vacancy_attempt_rate` or `charge_attempt_rate`/`electron_attempt_rate`, not both."))

    vacancy_rate_arg = isnothing(vacancy_attempt_rate) ? 1 : vacancy_attempt_rate
    carrier_rate_arg = isnothing(carrier_rate_arg) ? 1 : carrier_rate_arg

    vacancy_rate = _defect_internal_value(vacancy_rate_arg, physicalunits(role = :dimensionless), g, :vacancy_attempt_rate)
    carrier_rate = _defect_internal_value(carrier_rate_arg, physicalunits(role = :dimensionless), g, :charge_attempt_rate)
    (vacancy_rate >= zero(vacancy_rate) && carrier_rate >= zero(carrier_rate)) ||
        throw(ArgumentError("DefectsModel attempt rates must be nonnegative."))
    (vacancy_rate > zero(vacancy_rate) || carrier_rate > zero(carrier_rate)) ||
        throw(ArgumentError("DefectsModel requires at least one nonzero attempt rate."))
    return vacancy_rate, carrier_rate
end

"""
    DefectsModel(graph; vacancies, charges, ...)

Bind a two-species mobile charge model to `graph`. `vacancies` and `charges` may
be `MobileVacancies`/`MobileCharges` entries, integer counts for random
initialization, or explicit position collections when paired with the
corresponding `*_charge` and `*_hamiltonian` keywords.
"""
function DefectsModel(
    g::G;
    layer::L = 1,
    vacancies = nothing,
    charges = nothing,
    positive = nothing,
    negative = nothing,
    vacancy_charge::VC = 1,
    charge::CC = -1,
    positive_charge = vacancy_charge,
    negative_charge = charge,
    electron_charge = nothing,
    vacancy_hamiltonian = nothing,
    charge_hamiltonian = nothing,
    positive_effects = vacancy_hamiltonian,
    negative_effects = charge_hamiltonian,
    electron_hamiltonian = charge_hamiltonian,
    vacancy_attempt_rate = nothing,
    charge_attempt_rate = nothing,
    positive_attempt_rate = vacancy_attempt_rate,
    negative_attempt_rate = charge_attempt_rate,
    electron_attempt_rate = nothing,
    rng::R = Random.default_rng(),
) where {G<:AbstractIsingGraph,L,VC,CC,R<:AbstractRNG}
    vacancy_input = isnothing(vacancies) ? positive : vacancies
    charge_input = isnothing(charges) ? negative : charges
    isnothing(vacancy_input) &&
        throw(ArgumentError("DefectsModel requires a `vacancies` entry or legacy `positive` positions/count."))
    isnothing(charge_input) &&
        throw(ArgumentError("DefectsModel requires a `charges` entry or legacy `negative` positions/count."))
    vacancy_h = isnothing(vacancy_hamiltonian) ? positive_effects : vacancy_hamiltonian
    charge_h = isnothing(charge_hamiltonian) ? negative_effects : charge_hamiltonian
    carrier_charge = isnothing(electron_charge) ? negative_charge : electron_charge

    vacancy_entry = _mobile_species_from_input(
        vacancy_input;
        charge = positive_charge,
        hamiltonian = vacancy_h,
        constructor = MobileVacancies,
    )
    charge_entry = _mobile_species_from_input(
        charge_input;
        charge = carrier_charge,
        hamiltonian = isnothing(electron_hamiltonian) ? charge_h : electron_hamiltonian,
        constructor = MobileCharges,
    )

    vacancy_positions = _mobile_species_initial_positions(rng, g, layer, vacancy_entry, Int[])
    vacancy_system = DefectHopping(
        g;
        layer,
        defects = vacancy_positions,
        charge = vacancy_entry.charge,
        effects = _defect_species_hamiltonian(vacancy_entry),
    )
    charge_positions = _mobile_species_initial_positions(rng, g, layer, charge_entry, vacancy_system.defect_idxs)
    charge_system = DefectHopping(
        g;
        layer,
        defects = charge_positions,
        charge = charge_entry.charge,
        effects = _defect_species_hamiltonian(charge_entry),
    )
    vacancy_rate, carrier_rate = _mobile_attempt_rates(
        g;
        vacancy_attempt_rate = positive_attempt_rate,
        charge_attempt_rate = negative_attempt_rate,
        electron_attempt_rate,
    )
    charge_state = MobileChargeState(
        vacancy_system.defect_idxs,
        charge_system.defect_idxs,
        vacancy_system.occupancy,
        charge_system.occupancy,
    )
    return DefectsModel(vacancy_system, charge_system, charge_state, g, vacancy_rate, carrier_rate)
end

"""
    ChargeHopProposer(graph; kwargs...)

Compatibility constructor returning `DefectsModel(graph; kwargs...)`.
New code should call `DefectsModel` directly; `ChargeHopProposer(model)` is
reserved for deriving the actual proposal source from a defects model.
"""
function ChargeHopProposer(g::G; kwargs...) where {G<:AbstractIsingGraph}
    return DefectsModel(g; kwargs...)
end
