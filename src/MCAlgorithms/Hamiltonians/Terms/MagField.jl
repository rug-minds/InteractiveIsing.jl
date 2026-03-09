"""
H = Σ_i b_i s_i

The magnetic field part of the Ising Hamiltonian
"""
struct MagField{PV <: ParamTensor} <: HamiltonianTerm
    b::PV
end

@inline function MagField(; active = false, val = 0f0, homogeneous = false)
    if homogeneous
        return MagField(HomogeneousParam(val, 0; active, description = "Magnetic Field"))
    end
    return MagField(typeof(val), 0, active)
end

@inline MagField{PV}(g::AbstractIsingGraph) where PV <: ParamTensor = MagField{PV}(PV())

@inline HomogeneousMagField(g::AbstractIsingGraph, val = zero(eltype(g))) = reconstruct(MagField(active = true, val = val, homogeneous = true), g)
@inline MagField(g::AbstractIsingGraph, active::Bool) = reconstruct(MagField(active = active, val = zero(eltype(g))), g)
@inline MagField(type, len, active = false) = MagField(ParamTensor(zeros(type, len), type |> zero; active, description = "Magnetic Field"))

@inline function reconstruct(hterm::MagField, g::AbstractIsingGraph)
    T = eltype(g)
    len = statelen(g)

    if ishomogeneous(hterm.b)
        bnew = HomogeneousParam(
            convert(T, hterm.b[]),
            len;
            default = convert(T, default(hterm.b)),
            active = isactive(hterm.b),
            description = description(hterm.b),
        )
        return MagField(bnew)
    end

    vals = zeros(T, len)
    copylen = min(length(hterm.b), len)
    if copylen > 0
        @inbounds vals[1:copylen] .= convert.(T, hterm.b[1:copylen])
    end
    bnew = ParamTensor(
        vals,
        convert(T, default(hterm.b));
        active = isactive(hterm.b),
        description = description(hterm.b),
    )
    return MagField(bnew)
end

params(::Type{MagField}) = HamiltonianParams((:b, Vector{GraphType}, GraphType(0), "Magnetic Field"))

# function ΔH(::MagField, hargs, proposal)
@inline function calculate(::ΔH, hterm::MagField, state, proposal)
    j = at_idx(proposal)
    return -hterm.b[j]*(to_val(proposal) - state[j])
end

# function dH(::MagField, hargs, s_idx)
@inline function calculate(::dH, hterm::MagField, state, s_idx)
    return -hterm.b[s_idx]
end

 

# @ParameterRefs function deltaH(::MagField)
#     return (s[j] - sn[j])*b[j]
# end

# @NewParameterRefs function newDeltaH(args, ::MagField)
#     (;b) = args.hamiltonian
#     (;oldstate, newstate) = args

#     return (oldstate[j] - newstate[j])*b[j]
# end

# ΔH_paramrefs(::MagField) = @ParameterRefs b[j]*s[j]
# # H_expr(::Union{MagField, Type{<:MagField}}) = :(b[j]*s[j])
# ΔH_expr[MagField] = :(b[j]*s[j])
