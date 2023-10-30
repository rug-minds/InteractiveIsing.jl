function parseall(str)
    return Meta.parse("begin $str end").args
end

const clampexpr = "+ clampparam(g)/2.f0 * (newstate^2-oldstate^2 - 2f0 * @inbounds clamps(g)[idx] * (newstate-oldstate))"

function EdiffIsingExpr(stype::Type{ST}) where {ST <: SType}
    clamp = getSParam(stype, :Clamp)

    expr = "efac*(newstate-oldstate) $(clamp*clampexpr)"
    return Meta.parse(expr)
end

export EdiffIsingExpr


@inline @generated function EdiffIsing(g, stype::SType, idx, efac, oldstate, newstate)::Float32
    return EdiffIsingExpr(stype)
end
export EdiffIsing

