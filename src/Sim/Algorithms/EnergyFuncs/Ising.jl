abstract type ΔE end
abstract type dE end

@inline function ΔEIsing(g, oldstate, newstate, gstate, gadj, idx, @specialize(gstype), ::T) where T<:Union{Type{Discrete}, Type{Continuous}}
    ∂E = dEIsing(g, gstate, gadj, idx, gstype)
    return ΔEIsingClamp(g, ∂E, oldstate, newstate, gstype)
end

(ΔE)(::typeof(ΔEIsing)) = true
(dE)(::typeof(dEIsing)) = true