using StaticArrays
using LinearAlgebra

TwoVec = SVector{2,Float32}

abstract type PeriodicityType end
struct Periodic <: PeriodicityType end
struct NonPeriodic <: PeriodicityType end

struct LayerTopology
    layer::IsingLayer
    pvecs::Tuple{TwoVec, TwoVec}
    covecs::Tuple{TwoVec, TwoVec}
    periodic::Type{T} where T <: PeriodicityType

    function LayerTopology(layer, vec1::AbstractArray, vec2::AbstractArray, periodic = true)
        #calculation of covectors
        y1 = 1/(vec1[1]-(vec1[2]*vec2[1]/vec2[2]))
        x1 = 1/(vec1[2]-(vec1[1]*vec2[2]/vec2[1]))
        y2 = 1/(vec2[1]-(vec2[2]*vec1[1]/vec1[2]))
        x2 = 1/(vec2[2]-(vec2[1]*vec1[2]/vec1[1]))
        
        cov1 = TwoVec(y1, x1)
        cov2 = TwoVec(y2, x2)
    
        ptype = periodic ? Periodic : NonPeriodic
    
        return new(layer, (vec1, vec2), (cov1, cov2), ptype)
    end
end

@setterGetter LayerTopology

function (lt::LayerTopology)(y,x)
    point = TwoVec(y,x)
    comp1 = point ⋅ covecs(lt)[1]
    comp2 = point ⋅ covecs(lt)[2]

    return (comp1, comp2)
end

function latToPoint(layer, i::Integer, j::Integer)
    tp = top(layer)
    zag = i ÷ 2
    zig = i - zag
    zagvec = TwoVec(pvecs(tp)[1][1], -pvecs(tp)[1][2])
    return zig*pvecs(tp)[1] + zag*zagvec + j*pvecs(tp)[2]
end
export latToPoint