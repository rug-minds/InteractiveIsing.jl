"""
H = self[i]*s[i] 

The Quadratic self energy part of the Ising Hamiltonian
"""
struct Quadratic{S} <: HamiltonianTerm 
    self::S
end

Quadratic(g::AbstractIsingGraph) = Quadratic(g.self)

function calculate(::ΔH, hterm::Quadratic, state, proposal)
    j = at_idx(proposal)

    return hterm.self[j]*(to_val(proposal)^2 - state[j]^2)
end

function calculate(::dH, hterm::Quadratic, state, s_idx)

    return 2*hterm.self[s_idx]*state[s_idx]
end
