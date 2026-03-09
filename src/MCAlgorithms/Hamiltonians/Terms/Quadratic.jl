"""
H = self[i]*s[i] 

The Quadratic self energy part of the Ising Hamiltonian
"""
struct Quadratic{T, N} <: HamiltonianTerm 
    c::T
    self::AbstractArray{T,N}
end

Quadratic() = Quadratic(StaticFill(1.0), nothing)
Quadratic(g::AbstractIsingGraph) = reconstruct(Quadratic(), g)
reconstruct(::Quadratic, g::AbstractIsingGraph) = Quadratic(adj(g).diag)

function calculate(::ΔH, hterm::Quadratic, state, proposal)
    j = at_idx(proposal)

    return hterm.self[j]*(to_val(proposal)^2 - state[j]^2)
end

function calculate(::dH, hterm::Quadratic, state, s_idx)

    return 2*hterm.self[s_idx]*state[s_idx]
end
