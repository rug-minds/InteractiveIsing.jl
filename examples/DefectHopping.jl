struct DefectState{S,D,H}
    graphstate::S
    defectstate::D
    local_hamiltonian::H
end

# Local (Spin site) Hamiltonian Traits
localhamiltonian(::Quadratic) = true
localhamiltonian(::Quartic) = true
localhamiltonian(::Sextic) = true
localhamiltonian(::Hamiltonian) = false

struct DefectProposer <: AbstractProposer end
struct DefectProposal <: AbstractProposal
    proposedstate::DefectState
end


