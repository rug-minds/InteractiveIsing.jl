"""
E = 1/2 * sum_{i,j} w_{ij} s_i s_j
"""
struct Bilinear{A} <: HamiltonianTerm 
    adj::A
end

Bilinear(g::AbstractIsingGraph) = Bilinear(g.adj)

# function ΔH(::Bilinear, hargs, proposal)
function calculate(::ΔH, hterm::Bilinear, state, proposal)
    s = state
    wij = hterm.adj
    j = at_idx(proposal)
    cum = zero(eltype(s))
    @turbo for ptr in nzrange(wij, j)
        i = wij.rowval[ptr]
        w = wij.nzval[ptr]
        cum += w*s[i]
    end

    ising_energy = cum*(s[j] - to_val(proposal)) # s - to_val because of the - sign

    return ising_energy
end

# function dH(::Bilinear, hargs, s_idx)
function calculate(::dH, hterm::Bilinear, state, s_idx)
    s = state
    wij = hterm.adj
    cum = zero(eltype(s))
    @turbo for ptr in nzrange(wij, s_idx)
        i = wij.rowval[ptr]
        w = wij.nzval[ptr]
        cum += w*s[i]
    end
    ising_energy = -cum

    return ising_energy
end
