export Quartic
struct Quartic{PV <: ParamTensor} <: Hamiltonian 
    qc::PV
end

# Quartic holds a 0-dimensional (e.g. scalar) ParamTensor
Quartic(g::AbstractIsingGraph, val = 1) = Quartic(ScalarParam(eltype(g), val; description = "Quartic Coefficient"))

ΔH_expr[Quartic] = :(qc[]*self[j]*s[j]^4)