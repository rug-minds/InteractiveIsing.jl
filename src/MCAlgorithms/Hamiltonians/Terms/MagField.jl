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
function calculate(::ΔH, hterm::MagField, state, proposal)
    j = at_idx(proposal)
    return -hterm.b[j]*(to_val(proposal) - state[j])
end

# function dH(::MagField, hargs, s_idx)
function calculate(::dH, hterm::MagField, state, s_idx)
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

