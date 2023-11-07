abstract type ΔE end
abstract type dE end


# @inline function ΔEIsing(g, oldstate, newstate, gstate, gadj, idx, @specialize(gstype), ::Type{Discrete})
#     return @inbounds -2f0*oldstate*dEIsing(g, gstate, gadj, idx, gstype)
# end

# @inline function ΔEIsing(g, oldstate, newstate, gstate, gadj, idx, @specialize(gstype), ::Type{Discrete})
#     return @inbounds -2f0*oldstate*dEIsing(g, gstate, gadj, idx, gstype)
# end

@inline function ΔEIsing(g, oldstate, newstate, gstate, gadj, idx, @specialize(gstype), ::Type{Continuous})
    efactor = dEIsing(g, gstate, gadj, idx, gstype)
    return @inbounds efactor*(newstate-oldstate)
end
@inline function ΔEIsing(g, oldstate, newstate, gstate, gadj, idx, @specialize(gstype), ::Type{Discrete})
    efactor = dEIsing(g, gstate, gadj, idx, gstype)
    return @inbounds efactor*(newstate-oldstate)
end

(ΔE)(::typeof(ΔEIsing)) = true
(dE)(::typeof(dEIsing)) = true