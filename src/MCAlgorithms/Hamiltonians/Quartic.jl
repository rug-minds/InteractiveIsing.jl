export Quartic
struct Quartic{PV <: ParamVal} <: Hamiltonian 
    qc::PV
end

# Quartic holds a 0-dimensional (e.g. scalar) ParamVal
Quartic(g::AbstractIsingGraph, val = 1) = Quartic(ScalarParam(eltype(g), val; description = "Quartic Coefficient"))

ΔH_expr[Quartic] = :(qc[]*self[j]*s[j]^4)