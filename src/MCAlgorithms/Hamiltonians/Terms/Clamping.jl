export Clamping
"""
Clamping Hamiltonian for Equilibrium Propagation
H = β/2 *(s_i - y_i)^2

Where y_i is the target value for the i-th node
"""
struct Clamping <: Hamiltonian end
params(::Type{Clamping}, GraphType) = GatherHamiltonianParams((:β, GraphType, GraphType(0), "Clamping Factor"), (:y, Vector{GraphType}, GraphType(0), "Targets"))

function Δi_H(::Type{Clamping})
    collect_expr = :()
    return_expr = :(β*((sn_i^2-s_i^2)/2+y_i*(s_i-sn_i)))
    return HExpression(collect_expr, return_expr) 
end