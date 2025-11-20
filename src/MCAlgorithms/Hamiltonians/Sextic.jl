export Sextic
struct Sextic{PV <: ParamVal} <: Hamiltonian 
    sc::PV
end

# Sextic holds a 0-dimensional (e.g. scalar) ParamVal
Sextic(g::AbstractIsingGraph, val = 1) = Sextic(ScalarParam(eltype(g), val; description = "Sextic Coefficient"))

ΔH_expr[Sextic] = :(sc[]*self[j]*s[j]^6)