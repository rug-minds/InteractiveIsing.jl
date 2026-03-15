struct DefectState{D,S,H,N,F,T}
    defectstate::D
    graphstate::S
    size::NTuple{N,Int}
    local_hamiltonian::H
    charge::F
    top::T   
end

eltype(::Union{DefectState{D,S}, Type{<:DefectState{D,S}}}) where {D,S} = eltype(S)

# Local (Spin site) Hamiltonian Traits
localhamiltonian(::InteractiveIsing.Quadratic) = true
localhamiltonian(::Quartic) = true
localhamiltonian(::Sextic) = true
localhamiltonian(::InteractiveIsing.Hamiltonian) = false

power(::Quadratic) = 2
power(::Quartic) = 4
power(::Sextic) = 6

function DefectHamiltonian(hterms::HamiltonianTerms)
    h = InteractiveIsing.EmptyHamiltonian()
    for term in hterms.hs
        if localhamiltonian(term)
            h += term
        end
    end
    return h
end

function DefectState(g::IsingGraph, charge = 1.)
    @assert g isa SingleLayerG
    etype = eltype(g)
    charge = convert(etype, charge)
    defectstate = zeros(etype, size(state(g)))
    ham = DefectHamiltonian(graph(g).hamiltonian)
    return DefectState(defectstate, state(g), size(state(g)), ham, charge, top(g))
end

struct DefectProposer{DS,N,T} <: InteractiveIsing.AbstractProposer 
    defectstate::DS
    coordinates::MVector{N,Int}
    displacement::MVector{N,T}
end
struct DefectProposal{N,T} <: AbstractProposal
    coordinates::MVector{N,Int} where {N}
    displacement::MVector{N,T} where {N,T}
    accepted::Bool
end

Base.rand(proposer::DefectProposer) = rand(Rand.default_rng(), proposer)
function Base.rand(rng::AbstractRNG, proposer::DefectProposer)
    rand!(rng, (-1,1), proposer.displacement)
    rand!(rng, (0, size(proposer.defectstate) .- 1), proposer.coordinates)
    return DefectProposal(proposer.coordinates, proposer.displacement, false)
end



"""
Proposal to change the constants in the local potential terms
"""
struct LocalScaleProposal{T}
    change::T
    at_idx::Int
    accepted::Bool
end

function LocalScaleProposal(c::Real, accept = false)
    return LocalScaleProposal(convert(eltype(c), c), 0, accept)
end

@inline function calculate(::ΔH, hterm::Hamiltonian, isingstate::I, proposal::LocalScaleProposal) where I
    @assert localhamiltonian(hterm) "LocalScaleProposal can only be used with local Hamiltonian terms"
    ising_spin = isingstate[proposal.at_idx]
    c = hterm.c[]
    old_pot = hterm.lp[proposal.at_idx]
    new_pot = old_pot + proposal.change
    return  c*(new_pot-old_pot)^power(hterm)*ising_spin^power(hterm)
end

"""
Calculate spin energy in local potentials before and after the hop. 
"""
@inline function calculate(::ΔH, hamiltonian::H, state::DefectState, proposal::DefectProposal) where {H}
    coordinate_leaving = Coordinate(proposal.coordinates...)
    coordinate_entering = Coordinate(ntuple(i -> proposal.coordinates[i] + proposal.displacement[i], Val(length(proposal.coordinates))))

    if coordinate_entering in state.top
        # Valid hop, wrap coordinate and calculate energy change
        coordinate_entering = wrap(state.top,coordinate_entering)

        leaving_cproposal = LocalScaleProposal(-state.charge, CartesianIndex(proposal.coordinates), false)
        entering_cproposal = LocalScaleProposal(state.charge, CartesianIndex(proposal.coordinates .+ proposal.displacement), false)

        ΔE_leaving = @inline calculate(ΔH(), hamiltonian.local_hamiltonian, state, leaving_cproposal)
        ΔE_entering = @inline calculate(ΔH(), hamiltonian.local_hamiltonian, state, entering_cproposal)

        return ΔE_leaving + ΔE_entering
    else
        # Invalid hop, return infinite energy change to reject
        return eltype(Inf)
    end
end

accept(proposal::DefectProposal) = DefectProposal(proposal.coordinates, proposal.displacement, true)

@inline function accept(state::DefectState, f::DefectProposal)  
    prop = accept(f)
    state.defectstate[prop.coordinates] = 0
    state.defectstate[prop.coordinates .+ prop.displacement] = 1

    leaving_idx = CartesianIndex(prop.coordinates)
    entering_idx = CartesianIndex(prop.coordinates .+ prop.displacement)

    #Set the hamiltonian self terms
    for ham in state.local_hamiltonian
        ham.self[leaving_idx] -= state.charge
        ham.self[entering_idx] += state.charge
    end

    return state
end


