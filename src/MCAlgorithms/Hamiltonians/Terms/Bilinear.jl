"""
E = -1/2 * sum_{i,j} J_{ij} s_i s_j
"""
struct Bilinear{P} <: HamiltonianTerm
    parameters::P
end

_default_bilinear_adjacency(g) = adj(g)

@inline function Bilinear(; adj = nothing, J = nothing)
    isnothing(J) || isnothing(adj) ||
        throw(ArgumentError("Pass either `J` or `adj`, not both."))
    J = isnothing(J) ? adj : J
    params = Parameters(
        parameter(;
            J,
            type = AbstractMatrix,
            default = _default_bilinear_adjacency,
            ensure = ensure_isinggraph_adjacency,
            info = "Bilinear coupling matrix J_ij",
        ),
    )
    return Bilinear(params)
end

@inline Bilinear(g::AbstractIsingGraph) = instantiate(Bilinear(), g)

# function ΔH(::Bilinear, hargs, proposal)
@inline function calculate(::ΔH, hterm::BL, model, proposal) where {BL<:Bilinear}
    s = @inline graphstate(model)
    J = hterm.J
    j = at_idx(proposal)
    total = @inline weighted_neighbors_sum(j, J, s)
    ising_energy = total*(s[j] - to_val(proposal)) # s - s' because of the - sign

    return ising_energy
end

"""
    calculate(d_iH(), hterm::Bilinear, model, proposal)

Return the bilinear derivative for the spin identified by `proposal`, evaluated
at the proposal endpoint state.
"""
@inline function calculate(::d_iH, hterm::Bilinear, model, proposal::SingleSpinProposal)
    s = @inline graphstate(model)
    J = hterm.J
    spin_idx = @inline at_idx(proposal)
    total = @inline weighted_neighbors_sum(spin_idx, J, s)
    ising_energy = -total
    return ising_energy
end

@inline function parameter_derivative(hterm::Bilinear, state::S; dJ = similar(hterm.J), buffermode::BufferMode = OverwriteBuffer()) where {S <: AbstractArray}
    s = @inline state
    n = length(s)
    indexes = index_pairs_iterator(hterm.J, false)
    if buffermode isa OverwriteBuffer  
        for (ptr, (i,j)) in enumerate(indexes)
            dJ[ptr] = -1/2 * s[i] * s[j]
        end
    else
        for (ptr, (i,j)) in enumerate(indexes)
            dJ[ptr] += sign(buffermode) * -1/2 * s[i] * s[j]
        end
    end
    return (; dJ)
end
