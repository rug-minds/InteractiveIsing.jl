abstract type Hamiltonian end
abstract type HamiltonianTerm <: Hamiltonian end

abstract type AbstractHamiltonianTerms{HS} <: Hamiltonian end
# Base.pairs(ht::AbstractHamiltonianTerms) = pairs(merge(pairs.(getfield(ht, :hs))...))
getHS(::Type{<:AbstractHamiltonianTerms{HS}}) where {HS} = HS
getHS(::AbstractHamiltonianTerms{HS}) where {HS} = HS
getHS(h::Type{<:Hamiltonian}) = (h,)

"""
    parameter_derivative(hamiltonian, state, args...)

Return the derivative of a Hamiltonian or Hamiltonian term with respect to all of its
trainable parameters in a structure that mirrors the parameter tree you want to expose to
Lux or a custom `rrule`.
"""
function parameter_derivative end
