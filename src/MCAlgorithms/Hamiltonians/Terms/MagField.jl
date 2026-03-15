"""
H = -Σ_i b_i s_i

The magnetic field part of the Ising Hamiltonian
"""
Base.@kwdef struct MagField{PV} <: HamiltonianTerm
    b::PV = StateLike(ConstFill, 0)
end

# @inline function MagField(; b = StateLike(ConstFill, 0))
#     return MagField(b)
# end

@inline function reconstruct(hterm::MF, g::AbstractIsingGraph) where {MF <: MagField}
    T = eltype(g)
    len = statelen(g)
    b = nothing
    # @show hterm.b
    if hterm.b isa Function || hterm.b isa DerivedParameter
        b = hterm.b(g)
    else
        b = hterm.b
        @assert length(b) == len "Length of b must match number of spins in graph"
    end
    MagField(map(T, b))
end

@inline function calculate(::ΔH, hterm::MagField, state::S, proposal) where {S <: AbstractIsingGraph}
    j = at_idx(proposal)
    spins = @inline InteractiveIsing.state(state)
    return -hterm.b[j]*(to_val(proposal) - spins[j])
end

# function dH(::MagField, hargs, s_idx)
@inline function calculate(::dH, hterm::MagField, state::S, s_idx) where {S <: AbstractIsingGraph}
    return -hterm.b[s_idx]
end

