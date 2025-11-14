export Quartic
struct Quartic{PV <: ParamVal} <: Hamiltonian 
    qc::PV
end

Quartic(g::AbstractIsingGraph, val = 1) = Quartic(ParamVal(eltype(g)(val), val, true))

ΔH_expr[Quartic] = :(qc[]*self[j]*s[j]^4)