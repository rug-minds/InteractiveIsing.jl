# Previous non-template sketch kept here while the template interface settles.
#=
"""
H = -Σ_i b_i s_i

The magnetic field part of the Ising Hamiltonian
"""
Base.@kwdef struct MagField{PV} <: HamiltonianTerm
    # LEGACY / DEPRECATED:
    # StateLike is kept only in this commented reference implementation. The
    # active template version below uses `default = ConstFill(0)` plus generic
    # ensure functions instead.
    b::PV = StateLike(ConstFill, 0)
end

@inline function instantiate(hterm::MF, g::AbstractIsingGraph) where {MF <: MagField}
    T = eltype(g)
    len = statelen(g)
    b = nothing
    if hterm.b isa Function || hterm.b isa DerivedParameter
        b = hterm.b(g)
    else
        b = hterm.b
        @assert length(b) == len "Length of b must match number of spins in graph"
    end
    MagField(map(T, b))
end

@inline calculate(::H, hterm::MagField, model::S) where S <: AbstractIsingGraph = calculate(H(), hterm, model; b = hterm.b)
@inline function calculate(::H, ::MagField, model::S; b) where S
    s = @inline graphstate(model)
    return -dot(b, s)
end

@inline function calculate(::ΔH, hterm::MagField, model::S, proposal) where {S <: AbstractIsingGraph}
    j = at_idx(proposal)
    spins = @inline graphstate(model)
    return -hterm.b[j]*(to_val(proposal) - spins[j])
end

@inline function calculate(::d_iH, hterm::MagField, model::S, s_idx) where {S <: AbstractIsingGraph}
    return -hterm.b[s_idx]
end

@inline function parameter_derivative(hterm::MagField, state::S; db = similar(hterm.b), buffermode::BufferMode = OverwriteBuffer()) where {S <: AbstractArray}
    if buffermode isa OverwriteBuffer
        db .= -state
    else
        db .+= sign(buffermode) * -state
    end
    return (; db = db)
end
=#

"""
H = -Σ_i b_i s_i

The magnetic field part of the Ising Hamiltonian, written against the
term-template parameter convention.
"""
struct MagField{P} <: LocalPotential
    parameters::P
end

function MagField(; b = nothing, c = nothing)
    params = Parameters(
        parameter(;
            c,
            type = Number,
            default = 1,
            ensure = ensure_isinggraph_eltype,
            info = "Magnetic field coupling constant",
        ),
        parameter(;
            b,
            type = AbstractArray,
            default = ConstFill(0),
            default_type = UniformArray,
            ensure = (ensure_isinggraph_state_length, ensure_isinggraph_eltype),
            info = "Local magnetic field values b_i for each spin i",
        ),
    )
    return MagField(params)
end

@inline function calculate(::H, hterm::MagField, model)
    s = @inline graphstate(model)
    return -hterm.c * dot(hterm.b, s)
end

@inline function calculate(::ΔH, hterm::MagField, model, proposal)
    j = at_idx(proposal)
    spins = @inline graphstate(model)
    return -hterm.c * hterm.b[j] * (to_val(proposal) - spins[j])
end

@inline function calculate(::d_iH, hterm::MagField, model, s_idx)
    return -hterm.c * hterm.b[s_idx]
end

@inline function parameter_derivative(hterm::MagField, state::S; db = similar(hterm.b), buffermode::BufferMode = OverwriteBuffer()) where {S <: AbstractArray}
    if buffermode isa OverwriteBuffer
            db .= -hterm.c .* state
    else
        db .+= sign(buffermode) .* -hterm.c .* state
    end
    return (; db = db)
end
