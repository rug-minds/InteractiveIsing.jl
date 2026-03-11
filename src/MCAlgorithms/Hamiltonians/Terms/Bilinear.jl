"""
E = 1/2 * sum_{i,j} w_{ij} s_i s_j
"""
struct Bilinear{A} <: HamiltonianTerm 
    adj::A
end

@inline Bilinear() = Bilinear(nothing)
@inline Bilinear(g::AbstractIsingGraph) = reconstruct(Bilinear(), g)
@inline reconstruct(::Bilinear, g::AbstractIsingGraph) = Bilinear(adj(g))

# function ΔH(::Bilinear, hargs, proposal)
@inline function calculate(::ΔH, hterm::BL, state::S, proposal) where {BL<:Bilinear, S <: AbstractIsingGraph}
    s = @inline InteractiveIsing.state(state)
    wij = hterm.adj
    j = at_idx(proposal)
    total = @inline weighted_neighbors_sum(j, wij, s)
    ising_energy = total*(s[j] - to_val(proposal)) # s - s' because of the - sign

    return ising_energy
end

# function dH(::Bilinear, hargs, s_idx)
@inline function calculate(::dH, hterm::Bilinear, state::S, s_idx) where {S <: AbstractIsingGraph}
    s = @inline InteractiveIsing.state(state)
    wij = hterm.adj
    cum = zero(eltype(s))
    rowval = SparseArrays.getrowval(wij)
    nzval = SparseArrays.getnzval(wij)
    @turbo for ptr in nzrange(wij, s_idx)
        i = rowval[ptr]
        w = nzval[ptr]
        cum += w*s[i]
    end
    ising_energy = -cum

    return ising_energy
end
