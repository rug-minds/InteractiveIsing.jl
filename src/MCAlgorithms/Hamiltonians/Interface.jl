abstract type Hamiltonian end
abstract type HamiltonianTerm <: Hamiltonian end

abstract type AbstractHamiltonianTerms{HS} <: Hamiltonian end
# Base.pairs(ht::AbstractHamiltonianTerms) = pairs(merge(pairs.(getfield(ht, :hs))...))
getHS(::Type{<:AbstractHamiltonianTerms{HS}}) where {HS} = HS
getHS(::AbstractHamiltonianTerms{HS}) where {HS} = HS
getHS(h::Type{<:Hamiltonian}) = (h,)