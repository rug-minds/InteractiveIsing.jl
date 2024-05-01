function parseall(str)
    return Meta.parse("begin $str end").args
end

const clampexpr = "+ clampparam(g)/T(2) * (newstate^2-oldstate^2 - T(2) * @inbounds clamps(g)[idx] * (newstate-oldstate))"

@inline function ΔEIsingClampExpr(stype::Type{ST}) where {ST <: SType}
    clamp = getSParam(stype, :Clamp)

    expr = "dE*(newstate-oldstate) $(clamp*clampexpr)"
    return Meta.parse(expr)
end

export ΔEIsingClampExpr


@inline @generated function ΔEIsingClamp(g::IsingGraph{T}, dE, oldstate, newstate, stype::SType,)::Float32 where T
    return ΔEIsingClampExpr(stype)
end
export ΔEIsingClamp

