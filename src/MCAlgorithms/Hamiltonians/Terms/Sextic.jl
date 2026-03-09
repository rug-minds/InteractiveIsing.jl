export Sextic
struct Sextic{S, PV} <: HamiltonianTerm 
    self::S
    c::PV
end

@inline Sextic(val::Real = 1; self = nothing) = Sextic(self, FillArray(val))

# Sextic holds a 0-dimensional (e.g. scalar) ParamTensor
@inline Sextic(g::AbstractIsingGraph, val = 1) = reconstruct(Sextic(val), g)
@inline function reconstruct(hterm::Sextic, g::AbstractIsingGraph)
    T = eltype(g)
    c = map(eltype(g), hterm.c)
    if isnothing(hterm.self)
        return Sextic(g.adj.diag, c)
    elseif hterm.self == :homogeneous
        return Sextic(FillArray(one(T), size(g.adj.diag)), c)
    end
    # c = ParamTensor(
    #     fill(convert(T, hterm.c[])),
    #     convert(T, default(hterm.c));
    #     active = isactive(hterm.c),
    #     size = tuple(),
    #     description = description(hterm.c),
    # )
    # return Sextic(g.adj.diag, c)
end

# ΔH_expr[Sextic] = :(sc[]*self[j]*s[j]^6)
# @inline @Auto_ΔH function ΔH(::Sextic, hargs, proposal)
#     return :(sc[]*self[j]*s[j]^6)
# end

# function ΔH(::Sextic, hargs, proposal)
@inline function calculate(::ΔH, hterm::Sextic, state, proposal)
    j = at_idx(proposal)
    return hterm.c[]*hterm.self[j]*(to_val(proposal)^6 - state[j]^6)
end

# function dH(::Sextic, hargs, s_idx)
@inline function calculate(::dH, hterm::Sextic, state, s_idx)
    return 6*hterm.c[]*hterm.self[s_idx]*state[s_idx]^5
end
