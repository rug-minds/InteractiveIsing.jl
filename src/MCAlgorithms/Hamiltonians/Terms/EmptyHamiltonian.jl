"""
For initialization
"""
struct EmptyHamiltonian <: HamiltonianTerm end

@inline reconstruct(ham::Hamiltonian, g::AbstractIsingGraph) = ham
@inline reconstruct(::EmptyHamiltonian, g::AbstractIsingGraph) = EmptyHamiltonian()

export reconstruct
