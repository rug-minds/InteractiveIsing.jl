"""
H = Σ_ij J_ij s_i s_j

The Quadratic part of the Ising Hamiltonian
"""
struct Quadratic <: HamiltonianTerm end

Quadratic(type, len) = Quadratic()

# function ΔH(::Quadratic, hargs, proposal)
function calculate(::ΔH, hterm::Quadratic, hargs, proposal)
    s = hargs.s
    wij = hargs.w
    self = hargs.self
    j = at_idx(proposal)
    cum = zero(eltype(s))
    @turbo for ptr in nzrange(wij, j)
        i = wij.rowval[ptr]
        w = wij.nzval[ptr]
        cum += w*s[i]
    end

    ising_energy = cum*(s[j] - to_val(proposal)) # s - to_val because of the - sign

    return ising_energy + self[j]*(to_val(proposal)^2 - s[j]^2)

end

# function dH(::Quadratic, hargs, s_idx)
function calculate(::dH, hterm::Quadratic, hargs, s_idx)
    s = hargs.s
    wij = hargs.w
    self = hargs.self
    cum = zero(eltype(s))
    @turbo for ptr in nzrange(wij, s_idx)
        i = wij.rowval[ptr]
        w = wij.nzval[ptr]
        cum += w*s[i]
    end
    ising_energy = -cum

    return ising_energy + 2*self[s_idx]*s[s_idx]
end
