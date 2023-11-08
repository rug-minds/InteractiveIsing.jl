function parseall(str)
    return Meta.parse("begin $str end").args
end

const clampexpr = "+ clampparam(g)/T(2) * (newstate^2-oldstate^2 - T(2) * @inbounds clamps(g)[idx] * (newstate-oldstate))"

function EdiffIsingExpr(stype::Type{ST}) where {ST <: SType}
    clamp = getSParam(stype, :Clamp)

    expr = "efac*(newstate-oldstate) $(clamp*clampexpr)"
    return Meta.parse(expr)
end

export EdiffIsingExpr


@inline @generated function EdiffIsing(g::IsingGraph{T}, stype::SType, idx, efac, oldstate, newstate)::Float32 where T
    return EdiffIsingExpr(stype)
end
export EdiffIsing

