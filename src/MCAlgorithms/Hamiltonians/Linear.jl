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
    return (s_i*w_ij)*(s_j-sn_j) + (sn_j^2-s_j^2)*self_j
end