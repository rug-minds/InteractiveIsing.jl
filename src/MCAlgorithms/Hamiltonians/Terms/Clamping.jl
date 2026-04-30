export Clamping

"""
Clamping Hamiltonian for Equilibrium Propagation
H = β/2 *(s_i - y_i)^2

Where y_i is the target value for the i-th node
"""
struct Clamping{P} <: HamiltonianTerm
    parameters::P
end

@inline function Clamping(; β = 1f0, y = nothing)
    params = Parameters(
        parameter(;
            β,
            type = AbstractArray,
            default = ConstVal(1f0),
            ensure = ensure_isinggraph_scalar,
            info = "Clamping strength β",
        ),
        parameter(;
            y,
            type = AbstractVector,
            default = 0,
            ensure = ensure_isinggraph_state_vector,
            info = "Target state y_i",
        ),
    )
    return Clamping(params)
end

@inline Clamping(β::Real, y = nothing) = Clamping(; β, y)
@inline Clamping(β::AbstractArray, y = nothing) = Clamping(; β, y)
@inline Clamping(β::NoEnsure, y = nothing) = Clamping(; β, y)
@inline Clamping(β::Force, y = nothing) = Clamping(; β, y)

@inline function Clamping(g::AbstractIsingGraph, β = one(eltype(g)), y = nothing)
    return instantiate(Clamping(β, y), g)
end

params(::Type{Clamping}, GraphType) = GatherHamiltonianParams((:β, GraphType, GraphType(0), "Clamping Factor"), (:y, Vector{GraphType}, GraphType(0), "Targets"))


# function ΔH(::Clamping, hargs, proposal)
@inline function calculate(::ΔH, hterm::Clamping, model::S, proposal) where {S <: AbstractIsingGraph}
    j = at_idx(proposal)
    newstate = to_val(proposal)
    spins = @inline graphstate(model)
    return hterm.β[]/2*(newstate^2 - spins[j]^2 - 2*hterm.y[j]*(newstate - spins[j]))
end

@inline function calculate(::d_iH, hterm::Clamping, model::S, s_idx) where {S <: AbstractIsingGraph}
    spins = @inline graphstate(model)
    return hterm.β[]*(spins[s_idx] - hterm.y[s_idx])
end

function clamp!(c::Clamping, layer::AbstractIsingLayer, vals::V) where {V <: AbstractVector}
    @assert length(vals) == length(state(layer)) "Length of vals must match number of states in layer"
    c.y .= vals
    return
end
