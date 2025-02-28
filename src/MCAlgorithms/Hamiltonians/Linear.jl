"""
H = Σ_ij J_ij s_i s_j

The linear part of the Ising Hamiltonian
"""
struct Linear <: Hamiltonian end

params(::Linear) = nothing

function Δi_H(::Linear)
    return 0
end