export Quartic
struct Quartic{PV <: ParamTensor} <: HamiltonianTerm 
    qc::PV
end

# Quartic holds a 0-dimensional (e.g. scalar) ParamTensor
Quartic(g::AbstractIsingGraph, val = 1) = Quartic(ScalarParam(eltype(g), val; description = "Quartic Coefficient"))

@inline @Auto_ΔH function ΔH(::Quartic, hargs, proposal)
    return :(-qc[]*self[j]*s[j]^4)
end
