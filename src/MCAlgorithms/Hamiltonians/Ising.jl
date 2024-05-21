#TODO: move export Δi_H
export Ising, Δi_H

struct Ising <: Hamiltonian end
params(::Type{Ising}, GraphType::Type)  = GatherHamiltonianParams((:b, Vector{GraphType}, GraphType(0), "Magnetic Field"))

function Δi_H(::Type{Ising})
    collect_expr = :(w_ij*s_j)
    return_expr = :((s_i^2-sn_i^2)*self_i+(s_i-sn_i)*(b_i+collect_expr))
    return HExpression(collect_expr, return_expr)
end

