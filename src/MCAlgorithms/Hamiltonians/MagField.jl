"""
H = Σ_i b_i s_i

The magnetic field part of the Ising Hamiltonian
"""
struct MagField <: Hamiltonian end

params(::Type{MagField}) = HamiltonianParams((:b, Vector{GraphType}, GraphType(0), "Magnetic Field"))