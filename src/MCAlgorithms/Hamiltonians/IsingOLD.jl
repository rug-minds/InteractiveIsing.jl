# #TODO: move export Δi_H
# export IsingOLD, Δi_H

# struct IsingOLD <: Hamiltonian end

# params(::IsingOLD, GraphType::Type)  = GatherHamiltonianParams((:b, Vector{GraphType}, GraphType(0), "Magnetic Field"))

# function Δi_H(::IsingOLD)
#     collect_expr = :(w_ij*s_j)
#     return_expr = :((s_i^2-sn_i^2)*self_i+(s_i-sn_i)*(b_i+collect_expr))
#     return (;collect_expr, return_expr)
# end

# @ParameterRefs function deltaH(::IsingOLD)
#     return (s_i*w_ij)*(sn_j-s_j) + (s_j^2-sn_j^2)*self_j+(s_j-sn_j)*(b_j)
# end

# deltaH(ch::CompositeHamiltonian) = reduce(+, deltaH.(ch))