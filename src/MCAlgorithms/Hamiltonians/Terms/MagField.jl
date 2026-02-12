"""
H = Σ_i b_i s_i

The magnetic field part of the Ising Hamiltonian
"""
struct MagField{PV <: ParamTensor} <: Hamiltonian 
    b::PV
end

MagField{PV}(g::AbstractIsingGraph) where PV <: ParamTensor = MagField{PV}(PV())

MagField(g::AbstractIsingGraph) = MagField(eltype(g), statelen(g))
MagField(type, len, active = false) = MagField(ParamTensor(zeros(type, len), type |> zero; active, description = "Magnetic Field"))

params(::Type{MagField}) = HamiltonianParams((:b, Vector{GraphType}, GraphType(0), "Magnetic Field"))

@ParameterRefs function deltaH(::MagField)
    return (s[j] - sn[j])*b[j]
end

@NewParameterRefs function newDeltaH(args, ::MagField)
    (;b) = args.hamiltonian
    (;oldstate, newstate) = args

    return (oldstate[j] - newstate[j])*b[j]
end

ΔH_paramrefs(::MagField) = @ParameterRefs b[j]*s[j]
# H_expr(::Union{MagField, Type{<:MagField}}) = :(b[j]*s[j])
ΔH_expr[MagField] = :(b[j]*s[j])


