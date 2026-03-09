export Sextic
struct Sextic{S, PV <: ParamTensor} <: HamiltonianTerm 
    self::S
    c::PV
end

Sextic(val::Real = 1) = Sextic(nothing, ScalarParam(val; description = "Sextic Coefficient"))

# Sextic holds a 0-dimensional (e.g. scalar) ParamTensor
Sextic(g::AbstractIsingGraph, val = 1) = reconstruct(Sextic(val), g)
function reconstruct(hterm::Sextic, g::AbstractIsingGraph)
    T = eltype(g)
    c = ParamTensor(
        fill(convert(T, hterm.c[])),
        convert(T, default(hterm.c));
        active = isactive(hterm.c),
        size = tuple(),
        description = description(hterm.c),
    )
    return Sextic(g.adj.diag, c)
end

# ΔH_expr[Sextic] = :(sc[]*self[j]*s[j]^6)
# @inline @Auto_ΔH function ΔH(::Sextic, hargs, proposal)
#     return :(sc[]*self[j]*s[j]^6)
# end

# function ΔH(::Sextic, hargs, proposal)
function calculate(::ΔH, hterm::Sextic, state, proposal)
    j = at_idx(proposal)
    return hterm.sc[]*hterm.self[j]*(to_val(proposal)^6 - state[j]^6)
end

# function dH(::Sextic, hargs, s_idx)
function calculate(::dH, hterm::Sextic, state, s_idx)
    return 6*hterm.sc[]*hterm.self[s_idx]*state[s_idx]^5
end
