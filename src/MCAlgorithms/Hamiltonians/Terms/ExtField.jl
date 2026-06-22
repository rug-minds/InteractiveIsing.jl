"""
H = -Σ_i b_i s_i

The external-field part of the Ising Hamiltonian, written against the
term-template parameter convention.
"""
struct ExtField{P} <: LocalPotential
    parameters::P
end

function ExtField(; b = nothing, c = nothing)
    params = Parameters(
        parameter(;
            c,
            type = Number,
            default = 1,
            ensure = ensure_isinggraph_eltype,
            info = "External field coupling constant",
            units = physicalunits(role = :dimensionless),
        ),
        parameter(;
            b,
            type = AbstractArray,
            default = ConstFill(0),
            default_type = UniformArray,
            ensure = (ensure_isinggraph_state_length, ensure_isinggraph_eltype),
            info = "Local external field values b_i for each spin i",
            units = physicalunits(energy = 1, role = :field_energy),
        ),
    )
    return ExtField(params)
end

@inline function calculate(::H, hterm::ExtField, model)
    s = @inline graphstate(model)
    return -hterm.c * dot(hterm.b, s)
end

@inline function calculate(::ΔH, hterm::ExtField, model, proposal)
    j = at_idx(proposal)
    spins = @inline graphstate(model)
    return -hterm.c * hterm.b[j] * (to_val(proposal) - spins[j])
end

@inline function calculate(::d_iH, hterm::ExtField, model, proposal::SingleSpinProposal)
    s_idx = @inline at_idx(proposal)
    return -hterm.c * hterm.b[s_idx]
end

@inline function parameter_derivative(hterm::ExtField, state::S; db = similar(hterm.b), buffermode::BufferMode = OverwriteBuffer()) where {S <: AbstractArray}
    if buffermode isa OverwriteBuffer
            db .= -hterm.c .* state
    else
        db .+= sign(buffermode) .* -hterm.c .* state
    end
    return (; db = db)
end
