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
    get_proposer(model::DefectsModel)

Derive the charge-hop proposer used by `Metropolis` from the self-contained
mobile charge model.
"""
function get_proposer(model::M) where {M<:DefectsModel}
    return ChargeHopProposer(model)
end

"""
    graphstate(defects::DefectHopping)

Return the spin state used by the graph that owns this defect system.
"""
@inline graphstate(defects::D) where {D<:DefectHopping} = graphstate(defects.state)

@inline Base.eltype(defects::D) where {D<:DefectHopping} = _defect_parameter_type(defects.state)
@inline temp(defects::D) where {D<:DefectHopping} = temp(defects.state)

"""
    state(model::DefectsModel)

Return the grouped vacancy/carrier state for a mobile charge-hopping model.
"""
@inline state(model::M) where {M<:DefectsModel} = model.charge_state

"""
    graphstate(model::DefectsModel)

Return the spin state of the graph coupled to a mobile charge-hopping model.
"""
@inline graphstate(model::M) where {M<:DefectsModel} = graphstate(model.graph)

@inline temp(model::M) where {M<:DefectsModel} = temp(model.graph)

function Base.getproperty(defects::D, name::Symbol) where {D<:DefectHopping}
    name === :hamiltonian && return getproperty(getfield(defects, :state), :hamiltonian)
    return getfield(defects, name)
end

function Base.getproperty(state::S, name::Symbol) where {S<:MobileChargeState}
    name === :positive_idxs && return getfield(state, :vacancy_idxs)
    name === :negative_idxs && return getfield(state, :charge_idxs)
    name === :positive_occupancy && return getfield(state, :vacancy_occupancy)
    name === :negative_occupancy && return getfield(state, :charge_occupancy)
    return getfield(state, name)
end

function Base.getproperty(model::M, name::Symbol) where {M<:DefectsModel}
    name === :hamiltonian && return getproperty(getfield(model, :graph), :hamiltonian)
    name === :state && return getfield(model, :charge_state)
    name === :positive && return getfield(model, :vacancies)
    name === :negative && return getfield(model, :charges)
    name === :positive_attempt_rate && return getfield(model, :vacancy_attempt_rate)
    name === :negative_attempt_rate && return getfield(model, :charge_attempt_rate)
    return getfield(model, name)
end

function Base.getproperty(proposer::P, name::Symbol) where {P<:ChargeHopProposer}
    name === :vacancies && return getfield(proposer, :model).vacancies
    name === :charges && return getfield(proposer, :model).charges
    name === :positive && return getfield(proposer, :model).vacancies
    name === :negative && return getfield(proposer, :model).charges
    name === :vacancy_attempt_rate && return getfield(proposer, :model).vacancy_attempt_rate
    name === :charge_attempt_rate && return getfield(proposer, :model).charge_attempt_rate
    name === :positive_attempt_rate && return getfield(proposer, :model).vacancy_attempt_rate
    name === :negative_attempt_rate && return getfield(proposer, :model).charge_attempt_rate
    return getfield(proposer, name)
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

function init!(hts::HamiltonianTerms{Hs}, model::M) where {Hs,M<:DefectsModel}
    return init!(hts, model.graph)
end

function init!(hts::H, model::M) where {H<:Hamiltonian,M<:DefectsModel}
    return init!(hts, model.graph)
end
