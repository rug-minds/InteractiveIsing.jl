function parseall(str)
    return Meta.parse("begin $str end").args
end

# function Ediffstr(gtype, htype::HType{Params, Vals})::String where {Params, Vals}
#     str = " statediff = (newstate-oldstate)

#     efac*statediff"

#     extrastr = buildExpr(:DiffTerm, htype)

#     if extrastr != ""
#         str *= " + " * extrastr
#     end

#     return str
# end

const clampexpr = "+ clampparam(g)/2 * (newstate^2-oldstate^2 - 2 * @inbounds clamps(g)[idx] * (newstate-oldstate))"

function EdiffExpr(htype::HType{Params, Vals}) where {Params, Vals}
    clamp = getHParamType(typeof(htype), :Clamp)

    expr = "efac*(newstate-oldstate) $(clamp*clampexpr)"
    return Meta.parse(expr)
end

EdiffExpr(htype::Type{HType{Params, Vals}}) where {Params, Vals} = EdiffExpr(HType{Params, Vals}())

export EdiffExpr


@inline @generated function Ediff(g::T, htype::HType{Params, Vals}, idx, efac, oldstate, newstate)::Floet32 where {T, Params, Vals}
    return EdiffExpr(HType{Params, Vals}())
end
export Ediff

