"""
H = Σ_i b_i s_i

The magnetic field part of the Ising Hamiltonian
"""
struct MagField{PV <: ParamTensor} <: HamiltonianTerm
    b::PV
end

MagField{PV}(g::AbstractIsingGraph) where PV <: ParamTensor = MagField{PV}(PV())

HomogeneousMagField(g::AbstractIsingGraph, val = zero(eltype(g))) = MagField(HomogeneousParam(val, size(g)...; description = "Magnetic Field"))
MagField(g::AbstractIsingGraph, active::Bool) = MagField(eltype(g), statelen(g), active)
MagField(type, len, active = false) = MagField(ParamTensor(zeros(type, len), type |> zero; active, description = "Magnetic Field"))

params(::Type{MagField}) = HamiltonianParams((:b, Vector{GraphType}, GraphType(0), "Magnetic Field"))

# function ΔH(::MagField, hargs, proposal)
function calculate(::ΔH, hterm::MagField, hargs, proposal)
    s = hargs.s
    b = hargs.b
    j = at_idx(proposal)
    return -b[j]*(to_val(proposal) - s[j])
end

# function dH(::MagField, hargs, s_idx)
function calculate(::dH, hterm::MagField, hargs, s_idx)
    b = hargs.b
    return -b[s_idx]
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

