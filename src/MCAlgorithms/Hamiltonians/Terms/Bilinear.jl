"""
E = -1/2 * sum_{i,j} w_{ij} s_i s_j
"""
struct Bilinear{A} <: HamiltonianTerm 
    adj::A
end

@inline Bilinear(;adj = g -> adj(g)) = Bilinear(adj)
@inline Bilinear(g::AbstractIsingGraph) = reconstruct(Bilinear(), g)
@inline function reconstruct(b::Bilinear, g::AbstractIsingGraph)
    adj = nothing
    if b.adj isa Function
        adj = b.adj(g)
    else
        adj = b.adj
        @assert size(adj, 1) == length(graphstate(g)) && size(adj, 2) == length(graphstate(g)) "Adjacency matrix size must match number of spins in graph\nexpected $(length(graphstate(g)))x$(length(graphstate(g))), got $(size(adj))"
    end
    Bilinear(adj)
end

# function ΔH(::Bilinear, hargs, proposal)
@inline function calculate(::ΔH, hterm::BL, state::S, proposal) where {BL<:Bilinear, S <: AbstractIsingGraph}
    s = @inline graphstate(state)
    wij = hterm.adj
    j = at_idx(proposal)
    total = @inline weighted_neighbors_sum(j, wij, s)
    ising_energy = total*(s[j] - to_val(proposal)) # s - s' because of the - sign

    return ising_energy
end

# function d_iH(::Bilinear, hargs, s_idx)
@inline function calculate(::d_iH, hterm::Bilinear, state::S, s_idx) where {S <: AbstractIsingGraph}
    s = @inline graphstate(state)
    wij = hterm.adj
    total = @inline weighted_neighbors_sum(s_idx, wij, s)
    ising_energy = -total
    return ising_energy
end
