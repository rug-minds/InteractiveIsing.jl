function parseall(str)
    return Meta.parse("begin $str end").args
end

function Ediffstr(gtype, htype::HType{Params, Vals})::String where {Params, Vals}
    str = " statediff = (newstate-oldstate)

    efac*statediff"

    extrastr = buildExpr(:DiffTerm, htype)

    if extrastr != ""
        str *= " + " * extrastr
    end

    return str
end

Ediffstr(gtype, htype::Type{HType{Params, Vals}}) where {Params, Vals} = Ediffstr(gtype, HType{Params, Vals}())

export Ediffstr


@generated function Ediff(g::T, htype::HType{Params, Vals}, idx, efac, oldstate, newstate)::Floet32 where {T, Params, Vals}
    return Expr(:block, parseall(Ediffstr(T.parameters[1], htype))...)
end
export Ediff

