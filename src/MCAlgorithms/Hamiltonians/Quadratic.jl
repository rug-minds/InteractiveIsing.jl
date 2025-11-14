"""
H = Σ_ij J_ij s_i s_j

The Quadratic part of the Ising Hamiltonian
"""
struct Quadratic <: Hamiltonian end

Quadratic(type, len) = Quadratic()


ΔH_expr[Quadratic] = :(s[i]*w[i,j]*s[j] + self[j]*s[j]^2)



# @ParameterRefs function deltaH(::Quadratic)
#     return (s[i]*w[i,j])*(s[j]-sn[j]) + (sn[j]^2-s[j]^2)*self[j]
# end