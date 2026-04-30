"""
For initialization
"""
struct EmptyHamiltonian <: HamiltonianTerm end

Base.:+(::EmptyHamiltonian, h::HamiltonianTerm) = h
Base.:+(h::HamiltonianTerm, ::EmptyHamiltonian) = h

# @inline instantiate(ham::Hamiltonian, g::AbstractIsingGraph) = ham
@inline instantiate(::EmptyHamiltonian, g::AbstractIsingGraph) = EmptyHamiltonian()

export instantiate
