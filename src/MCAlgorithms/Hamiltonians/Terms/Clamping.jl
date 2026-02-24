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
Base.Expr(::Clamping) = :(-1/2*β*(s[j]^2)-y[j]*s[j])

params(::Type{Clamping}, GraphType) = GatherHamiltonianParams((:β, GraphType, GraphType(0), "Clamping Factor"), (:y, Vector{GraphType}, GraphType(0), "Targets"))


# function ΔH(::Clamping, hargs, proposal)
function calculate(::ΔH, hterm::Clamping, hargs, proposal)
    s = hargs.s
    β = hargs.β
    y = hargs.y
    j = at_idx(proposal)
    return -1/2*β[]*(to_val(proposal)^2 - s[j]^2) - y[j]*(to_val(proposal) - s[j])
end

function calculate(::dH, hterm::Clamping, hargs, s_idx)
    β = hargs.β
    y = hargs.y
    return -β[]*hargs.s[s_idx] - y[s_idx]
end