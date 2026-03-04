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
Clamping(g::AbstractIsingGraph, β = one(eltype(g)), y = zeros(eltype(g), nstatess(g))) = Clamping(Ref(β), y)

params(::Type{Clamping}, GraphType) = GatherHamiltonianParams((:β, GraphType, GraphType(0), "Clamping Factor"), (:y, Vector{GraphType}, GraphType(0), "Targets"))


# function ΔH(::Clamping, hargs, proposal)
function calculate(::ΔH, hterm::Clamping, state, proposal)
    j = at_idx(proposal)
    newstate = to_val(proposal)
    return hterm.β[]/2*(newstate^2 - state[j]^2 - 2*hterm.y[j]*(newstate - state[j]))
end

function calculate(::dH, hterm::Clamping, state, s_idx)
    return hterm.β[]*(state[s_idx] - hterm.y[s_idx])
end