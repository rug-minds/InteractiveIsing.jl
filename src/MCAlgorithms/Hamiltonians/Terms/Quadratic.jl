"""
H = self[i]*s[i] 

The Quadratic self energy part of the Ising Hamiltonian
"""
struct Quadratic{T, S} <: HamiltonianTerm 
    c::T
    self::S
end

Quadratic() = Quadratic(StaticFill(1.0), nothing)
function Quadratic(c; self = nothing)
    if isnothing(self)
        return Quadratic(StaticFill(c), nothing)
    else
        Quadratic(StaticFill(c), self)
    end
end
Quadratic(g::AbstractIsingGraph) = reconstruct(Quadratic(), g)
reconstruct(q::Quadratic, g::AbstractIsingGraph) = Quadratic(q.c, adj(g).diag)

function calculate(::ΔH, hterm::Quadratic, state, proposal)
    j = at_idx(proposal)

    return hterm.self[j]*(to_val(proposal)^2 - state[j]^2)
end

function calculate(::dH, hterm::Quadratic, state, s_idx)

    return 2*hterm.self[s_idx]*state[s_idx]
end
