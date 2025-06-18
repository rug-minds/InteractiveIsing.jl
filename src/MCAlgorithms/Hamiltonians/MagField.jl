"""
H = Σ_i b_i s_i

The magnetic field part of the Ising Hamiltonian
"""
struct MagField{PV <: ParamVal} <: Hamiltonian 
    b::PV
end

MagField{PV}(g::IsingGraph) where PV <: ParamVal = MagField{PV}(PV())

MagField(g) = MagField(eltype(g), statelen(g))z
MagField(type, len, active = false) = MagField(ParamVal(zeros(type, len), type |> zero, "Magnetic Field", active))

params(::Type{MagField}) = HamiltonianParams((:b, Vector{GraphType}, GraphType(0), "Magnetic Field"))

@ParameterRefs function deltaH(::MagField)
    return (s_j - sn_j)*b_j
end