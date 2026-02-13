"""
H = Σ_ij J_ij s_i s_j

The Quadratic part of the Ising Hamiltonian
"""
struct Quadratic <: HamiltonianTerm end

Quadratic(type, len) = Quadratic()

# ΔH_paramrefs(::Quadratic) = @ParameterRefs (s[i]*w[i,j]*s[j] + self[j]*s[j]^2)


@inline @Auto_ΔH function ΔH(::Quadratic, hargs, proposal)
    return :(-s[i]*w[i,j]*s[j] + self[j]*s[j]^2)
end
