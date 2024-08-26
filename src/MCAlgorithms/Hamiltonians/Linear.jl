"""
H = Σ_ij J_ij s_i s_j

The linear part of the Ising Hamiltonian
"""
struct Linear <: Hamiltonian 

params(::Type{Linear}) = nothing

function Δi_H(::Type{Linear})
    return nothing
end