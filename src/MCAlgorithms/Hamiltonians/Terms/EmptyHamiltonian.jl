"""
For initialization
"""
struct EmptyHamiltonian <: HamiltonianTerm end

Base.:+(::EmptyHamiltonian, h::HamiltonianTerm) = h
Base.:+(h::HamiltonianTerm, ::EmptyHamiltonian) = h

@inline reconstruct(ham::Hamiltonian, g::AbstractIsingGraph) = ham
@inline reconstruct(::EmptyHamiltonian, g::AbstractIsingGraph) = EmptyHamiltonian()

export reconstruct
