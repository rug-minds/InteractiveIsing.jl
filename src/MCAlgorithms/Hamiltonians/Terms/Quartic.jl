export Quartic
struct Quartic{S, PV} <: HamiltonianTerm
    self::S
    c::PV
end

@inline Quartic(val::Real = 1; self = nothing) = Quartic(self, FillArray(val))

# Quartic holds a 0-dimensional (e.g. scalar) ParamTensor
@inline Quartic(g::AbstractIsingGraph, val = 1) = reconstruct(Quartic(val), g)
@inline function reconstruct(hterm::Quartic, g::AbstractIsingGraph)
    T = eltype(g)
    c = map(eltype(g), hterm.c)
    if isnothing(hterm.self)
        return Quartic(g.adj.diag, c)
    elseif hterm.self == :homogeneous
        return Quartic(FillArray(one(T), size(g.adj.diag)), c)
    end
    # c = ParamTensor(
    #     fill(convert(T, hterm.c[])),
    #     convert(T, default(hterm.c));
    #     active = isactive(hterm.c),
    #     size = tuple(),
    #     description = description(hterm.c),
    # )
    # return Quartic(g.adj.diag, c)
end

# @inline @Auto_ΔH function ΔH(::Quartic, hargs, proposal)
#     return :(c[]*self[j]*s[j]^4)
# end

# function ΔH(::Quartic, hargs, proposal)
@inline function calculate(::ΔH, hterm::Quartic, state, proposal)
    j = at_idx(proposal)
    return hterm.c[]*hterm.self[j]*(to_val(proposal)^4 - state[j]^4)
end

# function dH(::Quartic, hargs, s_idx)
@inline function calculate(::dH, hterm::Quartic, state, s_idx)
    return 4*hterm.c[]*hterm.self[s_idx]*state[s_idx]^3
end
