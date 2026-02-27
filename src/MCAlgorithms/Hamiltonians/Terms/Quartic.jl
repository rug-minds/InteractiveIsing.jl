export Quartic
struct Quartic{S, PV <: ParamTensor} <: HamiltonianTerm
    self::S
    qc::PV
end

# Quartic holds a 0-dimensional (e.g. scalar) ParamTensor
Quartic(g::AbstractIsingGraph, val = 1) = Quartic(g.self, ScalarParam(eltype(g), val; description = "Quartic Coefficient"))

# @inline @Auto_ΔH function ΔH(::Quartic, hargs, proposal)
#     return :(qc[]*self[j]*s[j]^4)
# end

# function ΔH(::Quartic, hargs, proposal)
function calculate(::ΔH, hterm::Quartic, state, proposal)
    j = at_idx(proposal)
    return hterm.qc[]*hterm.self[j]*(to_val(proposal)^4 - state[j]^4)
end

# function dH(::Quartic, hargs, s_idx)
function calculate(::dH, hterm::Quartic, state, s_idx)
    return 4*hterm.qc[]*hterm.self[s_idx]*state[s_idx]^3
end