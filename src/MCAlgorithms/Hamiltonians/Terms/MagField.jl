"""
H = -Σ_i b_i s_i

The magnetic field part of the Ising Hamiltonian
"""
Base.@kwdef struct MagField{PV} <: HamiltonianTerm
    b::PV = StateLike(ConstFill, 0)
end



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


@inline calculate(::H, hterm::MagField, state::S) where S <: AbstractIsingGraph = calculate(H(), hterm, state; b = hterm.b)
@inline function calculate(::H, ::MagField, state::S; b) where S
    s = @inline graphstate(state)
    return -dot(b, s)
end


@inline function calculate(::ΔH, hterm::MagField, state::S, proposal) where {S <: AbstractIsingGraph}
    j = at_idx(proposal)
    spins = @inline graphstate(state)
    return -hterm.b[j]*(to_val(proposal) - spins[j])
end

# function d_iH(::MagField, hargs, s_idx)
@inline function calculate(::d_iH, hterm::MagField, state::S, s_idx) where {S <: AbstractIsingGraph}
    return -hterm.b[s_idx]
end

@inline function parameter_derivative(hterm::MagField, state::AbstractArray; db = similar(hterm.b), buffermode::BufferMode = OverwriteBuffer()) where {S <: AbstractIsingGraph}
    if buffermode isa OverwriteBuffer
        db .= -state
    else
        db .+= sign(buffermode) * -state
    end
    return (; db = db)
end
