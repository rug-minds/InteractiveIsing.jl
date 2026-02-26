struct DefectState{S,D,H}
    graphstate::S
    defectstate::D
    local_hamiltonian::H
end

# Local Hamiltonian Traits
localhamiltonian(::Quadratic) = true
localhamiltonian(::Quartic) = true
localhamiltonian(::Sextic) = true
localhamiltonian(::Hamiltonian) = false

struct HoppingProposer end
