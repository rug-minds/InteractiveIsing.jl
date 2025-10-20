"""
H = Σ_ij J_ij s_i s_j

The linear part of the Ising Hamiltonian
"""
struct Linear <: Hamiltonian end

Linear(type, len) = Linear()

# params(::Linear) = nothing

# function Δi_H(::Linear)
#     return 0
# end

@ParameterRefs function deltaH(::Linear)
    return (s[i]*w[i,j])*(s[j]-sn[j]) + (sn[j]^2-s[j]^2)*self[j]
end