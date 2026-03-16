export Clamping

"""
Clamping Hamiltonian for Equilibrium Propagation
H = β/2 *(s_i - y_i)^2

Where y_i is the target value for the i-th node
"""
struct Clamping{PBeta, BY} <: HamiltonianTerm
    β::PBeta
    y::BY
end

@inline Clamping(β::Real = 1f0, y::AbstractVector = Float32[]) = Clamping(Ref(β), y)
@inline Clamping(β::Real, y::Fill) = Clamping(Ref(β), y)
@inline function Clamping(g::AbstractIsingGraph, β = one(eltype(g)), y = nothing)
    isnothing(y) && (y = zeros(eltype(g), nstates(g)))
    return reconstruct(Clamping(β, y), g)
end
@inline function reconstruct(hterm::Clamping, g::AbstractIsingGraph)
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
@inline function calculate(::ΔH, hterm::Clamping, state::S, proposal) where {S <: AbstractIsingGraph}
    j = at_idx(proposal)
    newstate = to_val(proposal)
    spins = @inline InteractiveIsing.state(state)
    return hterm.β[]/2*(newstate^2 - spins[j]^2 - 2*hterm.y[j]*(newstate - spins[j]))
end

@inline function calculate(::dH, hterm::Clamping, state::S, s_idx) where {S <: AbstractIsingGraph}
    spins = @inline InteractiveIsing.state(state)
    return hterm.β[]*(spins[s_idx] - hterm.y[s_idx])
end
