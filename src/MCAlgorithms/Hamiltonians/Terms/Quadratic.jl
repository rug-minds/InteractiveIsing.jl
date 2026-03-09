"""
H = self[i]*s[i] 

The Quadratic self energy part of the Ising Hamiltonian
"""
struct Quadratic{T, S} <: HamiltonianTerm 
    c::T
    self::S
end

@inline Quadratic() = Quadratic(StaticFill(1.0), nothing)
@inline function Quadratic(c; self = nothing)
    if isnothing(self)
        return Quadratic(StaticFill(c), nothing)
    else
        Quadratic(StaticFill(c), self)
    end
end
@inline Quadratic(g::AbstractIsingGraph) = reconstruct(Quadratic(), g)
@inline reconstruct(q::Quadratic, g::AbstractIsingGraph) = Quadratic(map(eltype(g),q.c), adj(g).diag)

@inline function calculate(::ΔH, hterm::Q, state, proposal) where Q <: Quadratic
    j = at_idx(proposal)
    return hterm.c[]*hterm.self[j]*(to_val(proposal)^2 - state[j]^2)
end

@inline function calculate(::dH, hterm::Quadratic, state, s_idx)
    return 2*hterm.c[]*hterm.self[s_idx]*state[s_idx]
end
