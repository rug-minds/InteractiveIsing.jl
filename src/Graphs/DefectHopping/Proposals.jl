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

Draw one hop from either the vacancy or mobile charge field.
"""
function Base.rand(rng::R, proposer::P) where {R<:AbstractRNG,P<:ChargeHopProposer}
    model = proposer.model
    nvacancies = length(model.vacancies.defect_idxs)
    ncharges = length(model.charges.defect_idxs)
    vacancy_weight = model.vacancy_attempt_rate * nvacancies
    charge_weight = model.charge_attempt_rate * ncharges
    total_weight = vacancy_weight + charge_weight
    total_weight > zero(total_weight) ||
        throw(ArgumentError("ChargeHopProposer has no active proposal species; check attempt rates and occupancies."))

    if rand(rng) * total_weight < vacancy_weight
        return ChargeHopProposal(PositiveFreeCharge(), rand(rng, model.vacancies))
    else
        return ChargeHopProposal(NegativeFreeCharge(), rand(rng, model.charges))
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
    return ChargeHopProposal(proposal.species, accept(proposer.model.vacancies, proposal.proposal))
end

function accept(proposer::P, proposal::NP) where {P<:ChargeHopProposer,NP<:ChargeHopProposal{NegativeFreeCharge}}
    return ChargeHopProposal(proposal.species, accept(proposer.model.charges, proposal.proposal))
end

"""
    accept(model::DefectsModel, proposal::ChargeHopProposal)

Commit an accepted mobile charge proposal through the model-level API. This is
the direct analogue of accepting a proposal against an `IsingGraph`-derived
proposer, but keeps the self-contained charge model usable in tests and manual
code.
"""
function accept(model::M, proposal::P) where {M<:DefectsModel,P<:ChargeHopProposal}
    return accept(get_proposer(model), proposal)
end

@inline _charge_hop_system(model::M, proposal::P) where {M<:DefectsModel,P<:ChargeHopProposal{PositiveFreeCharge}} =
    model.vacancies

@inline _charge_hop_system(model::M, proposal::P) where {M<:DefectsModel,P<:ChargeHopProposal{NegativeFreeCharge}} =
    model.charges

"""
    update!(::Metropolis, hterm, model::DefectsModel, proposal::ChargeHopProposal)

Route accepted combined charge proposals to the selected vacancy or carrier
field's existing update methods.
"""
@inline function update!(algo::A, hterm::H, model::M, proposal::P) where {A<:Metropolis,H<:Hamiltonian,M<:DefectsModel,P<:ChargeHopProposal}
    return update!(algo, hterm, _charge_hop_system(model, proposal), proposal.proposal)
end

@inline function update!(algo::A, hts::HTS, model::M, proposal::P) where {A<:Metropolis,HTS<:AbstractHamiltonianTerms,M<:DefectsModel,P<:ChargeHopProposal}
    return update!(algo, hts, _charge_hop_system(model, proposal), proposal.proposal)
end

@inline function update!(algo::A, hts::HTS, model::M, proposal::P) where {A<:Metropolis,Hs,HTS<:HamiltonianTerms{Hs},M<:DefectsModel,P<:ChargeHopProposal}
    return update!(algo, hts, _charge_hop_system(model, proposal), proposal.proposal)
end

@inline function update!(algo::A, hterm::H, model::M, proposal::P) where {A<:Metropolis,H<:HamiltonianTerm,M<:DefectsModel,P<:ChargeHopProposal}
    return update!(algo, hterm, _charge_hop_system(model, proposal), proposal.proposal)
end
