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
Clamping(β::Real = 1f0, y::AbstractVector = Float32[]) = Clamping(Ref(β), y)
function Clamping(g::AbstractIsingGraph, β = one(eltype(g)), y = nothing)
    isnothing(y) && (y = zeros(eltype(g), nstates(g)))
    return reconstruct(Clamping(β, y), g)
end
function reconstruct(hterm::Clamping, g::AbstractIsingGraph)
    T = eltype(g)
    ynew = zeros(T, nstates(g))
    copylen = min(length(hterm.y), length(ynew))
    if copylen > 0
        @inbounds ynew[1:copylen] .= convert.(T, hterm.y[1:copylen])
    end
    return Clamping(Ref(convert(T, hterm.β[])), ynew)
end

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
