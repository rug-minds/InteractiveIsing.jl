"""
For initialization
"""
struct EmptyHamiltonian <: HamiltonianTerm end

reconstruct(ham::Hamiltonian, g::AbstractIsingGraph) = ham
reconstruct(::EmptyHamiltonian, g::AbstractIsingGraph) = EmptyHamiltonian()

export reconstruct
