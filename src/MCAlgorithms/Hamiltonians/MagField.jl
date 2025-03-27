"""
H = Σ_i b_i s_i

The magnetic field part of the Ising Hamiltonian
"""
struct MagField{PV <: ParamVal} <: Hamiltonian 
    b::PV
end

MagField{PV}(g::IsingGraph) where PV <: ParamVal = MagField{PV}(PV())
MagField(g) = MagField(ParamVal(zeros(eltype(g), length(state(g))), eltype(g) |> zero, "Magnetic Field", true))

params(::Type{MagField}) = HamiltonianParams((:b, Vector{GraphType}, GraphType(0), "Magnetic Field"))

@ParameterRefs function deltaH(::MagField)
    return (s_j - sn_j)*b_j
end