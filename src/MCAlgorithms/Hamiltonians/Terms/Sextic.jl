export Sextic
struct Sextic{PV <: ParamTensor} <: HamiltonianTerm 
    sc::PV
end

# Sextic holds a 0-dimensional (e.g. scalar) ParamTensor
Sextic(g::AbstractIsingGraph, val = 1) = Sextic(ScalarParam(eltype(g), val; description = "Sextic Coefficient"))

# ΔH_expr[Sextic] = :(sc[]*self[j]*s[j]^6)
# @inline @Auto_ΔH function ΔH(::Sextic, hargs, proposal)
#     return :(sc[]*self[j]*s[j]^6)
# end

# function ΔH(::Sextic, hargs, proposal)
function calculate(::ΔH, hterm::Sextic, hargs, proposal)
    s = hargs.s
    self = hargs.self
    j = at_idx(proposal)
    return hargs.sc[]*self[j]*(to_val(proposal)^6 - s[j]^6)
end

# function dH(::Sextic, hargs, s_idx)
function calculate(::dH, hterm::Sextic, hargs, s_idx)
    s = hargs.s
    self = hargs.self
    return 6*hargs.sc[]*self[s_idx]*s[s_idx]^5
end

