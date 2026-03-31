"""
E = -1/2 * sum_{i,j} J_{ij} s_i s_j
"""
struct Bilinear{A} <: HamiltonianTerm 
    J::A
end

@inline Bilinear(;adj = g -> adj(g)) = Bilinear(adj)
@inline Bilinear(g::AbstractIsingGraph) = reconstruct(Bilinear(), g)
@inline function reconstruct(b::Bilinear, g::AbstractIsingGraph)
    J = nothing
    if b.J isa Function
        J = b.J(g)
    else
        J = b.J
        @assert size(J, 1) == length(graphstate(g)) && size(J, 2) == length(graphstate(g)) "Adjacency matrix size must match number of spins in graph\nexpected $(length(graphstate(g)))x$(length(graphstate(g))), got $(size(J))"
    end
    Bilinear(J)
end

# function ΔH(::Bilinear, hargs, proposal)
@inline function calculate(::ΔH, hterm::BL, state::S, proposal) where {BL<:Bilinear, S <: AbstractIsingGraph}
    s = @inline graphstate(state)
    J = hterm.J
    j = at_idx(proposal)
    total = @inline weighted_neighbors_sum(j, J, s)
    ising_energy = total*(s[j] - to_val(proposal)) # s - s' because of the - sign

    return ising_energy
end

# function d_iH(::Bilinear, hargs, s_idx)
@inline function calculate(::d_iH, hterm::Bilinear, state::S, s_idx) where {S <: AbstractIsingGraph}
    s = @inline graphstate(state)
    J = hterm.J
    total = @inline weighted_neighbors_sum(s_idx, J, s)
    ising_energy = -total
    return ising_energy
end

@inline function parameter_derivative(hterm::Bilinear, state::AbstractVector; dJ = similar(hterm.J), buffermode::BufferMode = OverwriteBuffer()) where {S <: AbstractIsingGraph}
    s = @inline graphstate(state)
    n = length(s)
    indexes = index_pairs_iterator(hterm.J, false)
    if buffermode isa OverwriteBuffer  
        for (ptr, (i,j)) in enumerate(indexes)
            d_J[ptr] = -1/2 * s[i] * s[j]
        end
    else
        for (ptr, (i,j)) in enumerate(indexes)
            d_J[ptr] += sign(buffermode) * -1/2 * s[i] * s[j]
        end
    end
    return (; d_J)
end
