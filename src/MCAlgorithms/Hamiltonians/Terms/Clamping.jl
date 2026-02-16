export Clamping
"""
Clamping Hamiltonian for Equilibrium Propagation
H = β/2 *(s_i - y_i)^2

Where y_i is the target value for the i-th node
"""
struct Clamping{PBeta, BY} <: Hamiltonian 
    β::PBeta
    y::BY
end
Clamping(g::AbstractIsingGraph, β = 1) = Clamping(ScalarParam(eltype(g), β; description = "Clamping Factor"), VectorParam(eltype(g), length(g); description = "Targets"))

params(::Type{Clamping}, GraphType) = GatherHamiltonianParams((:β, GraphType, GraphType(0), "Clamping Factor"), (:y, Vector{GraphType}, GraphType(0), "Targets"))

@Auto_ΔH function ΔH(::Clamping, hargs, proposal)
    return :(-1/2*β[]*(s[j]^2)-y[j]*s[j])
end

# function Δi_H(::Type{Clamping})
#     collect_expr = :()
#     return_expr = :(β*((sn_i^2-s_i^2)/2+y_i*(s_i-sn_i)))
#     return HExpression(collect_expr, return_expr) 
# end

