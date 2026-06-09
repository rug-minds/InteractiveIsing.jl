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

@inline function calculate(::d_iH, hterm::MagField, model, proposal::SingleSpinProposal)
    s_idx = @inline at_idx(proposal)
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
