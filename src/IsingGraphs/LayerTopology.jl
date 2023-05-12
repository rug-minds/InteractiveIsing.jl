using StaticArrays
using LinearAlgebra

TwoVec = SVector{2,Float32}

abstract type PeriodicityType end
struct Periodic <: PeriodicityType end
struct NonPeriodic <: PeriodicityType end
export PeriodicityType, Periodic, NonPeriodic

abstract type LatticeType end
struct Square <: LatticeType end
struct Rectangular <: LatticeType end
struct Oblique <: LatticeType end
struct Hexagonal <: LatticeType end
struct Rhombic <: LatticeType end
struct AnyLattice <: LatticeType end
export LatticeType, Square, Rectangular, Oblique, Hexagonal, Rhombic, AnyLattice

struct LayerTopology
    layer::IsingLayer
    pvecs::Tuple{TwoVec, TwoVec}
    covecs::Tuple{TwoVec, TwoVec}
    periodic::Type{T} where T <: PeriodicityType
    type::Type{U} where U <: LatticeType

    function LayerTopology(layer, vec1::AbstractArray, vec2::AbstractArray; periodic = true)
        #calculation of covectors
        y1 = 1/(vec1[1]-(vec1[2]*vec2[1]/vec2[2]))
        x1 = 1/(vec1[2]-(vec1[1]*vec2[2]/vec2[1]))
        y2 = 1/(vec2[1]-(vec2[2]*vec1[1]/vec1[2]))
        x2 = 1/(vec2[2]-(vec2[1]*vec1[2]/vec1[1]))
        
        cov1 = TwoVec(y1, x1)
        cov2 = TwoVec(y2, x2)
    
        ptype = periodic ? Periodic : NonPeriodic

        lattice_type = AnyLattice

        if vec1 == [1,0] && vec2 == [0,1]
            lattice_type = Square
        end
            
    
        return new(layer, (vec1, vec2), (cov1, cov2), ptype, lattice_type)
    end
end
export LayerTopology

@setterGetter LayerTopology

function testdist(i1,j1,i2,j2,top)
    l = glength(layer(top))
    w = gwidth(layer(top))

    i1 = i1 > l ? i1 - l : i1
    i2 = i2 > l ? i2 - l : i2
    j1 = j1 > w ? j1 - w : j1
    j2 = j2 > w ? j2 - w : j2

    pos1 = pos(i1,j1,pvecs(top))
    pos2 = pos(i2,j2,pvecs(top))

    return sqrt(sum((pos1-pos2).^2))
end

function testdist(i1,j1,i2,j2,top, l, w)

    i1 = i1 > l ? i1 - l : i1
    i2 = i2 > l ? i2 - l : i2
    j1 = j1 > w ? j1 - w : j1
    j2 = j2 > w ? j2 - w : j2

    pos1 = pos(i1,j1,pvecs(top))
    pos2 = pos(i2,j2,pvecs(top))

    return sqrt(sum((pos1-pos2).^2))
end
export testdist

function pos(i,j, pvecs::Tuple{TwoVec, TwoVec})
    return i*pvecs[1] + j*pvecs[2]
end

pos(idx, top) = pos(idxToCoord(idx, glength(layer(top))), top)
export pos

function dist2(i1, j1, i2, j2; pt::Type{NonPeriodic}, lt::Type{T}, pvecs, l::Int32 = 1, w::Int32 = 1) where T <: Union{Square, AnyLattice}
    return sum((pos(i1,j1, pvecs) - pos(i2,j2, pvecs)).^2)
end

function dist2(i1, j1, i2, j2; pt::Type{Periodic}, lt::Type{T}, pvecs, l::Int32, w::Int32) where T <: Union{Square, AnyLattice}
    i1 = i1 > l ? i1 - l : i1
    i2 = i2 > l ? i2 - l : i2
    j1 = j1 > w ? j1 - w : j1
    j2 = j2 > w ? j2 - w : j2

    return sum((pos(i1,j1, pvecs) - pos(i2,j2, pvecs)).^2)
end

dist2(i1, j1, i2, j2, top::LayerTopology) = dist2(i1, j1, i2, j2, pt = periodic(top), lt = type(top), pvecs = pvecs(top), l = glength(layer(top)), w = gwidth(layer(top)))

dist(i1, j1, i2, j2; pt::Type{PT}, lt::Type{T}, pvecs::Tuple{TwoVec,TwoVec}, l = 1, w = 1) where {PT <: PeriodicityType,T <: Union{Square, AnyLattice}} = sqrt(dist2(i1, j1, i2, j2; pt, lt, pvecs, l, w))
dist(i1, j1, i2, j2, top::LayerTopology) = dist(i1, j1, i2, j2, pt = periodic(top), lt = type(top), pvecs = pvecs(top), l = glength(layer(top)), w = gwidth(layer(top)))
dist(idx1,idx2,top) = dist(idxToCoord(idx1, glength(layer(top))), idxToCoord(idx2, glength(layer(top))), top)
export dist2, dist

function getDistFunc(top::LayerTopology)
    periodicity = periodic(top)
    lattype = type(top)
    return (i1,j1,i2,j2) -> dist(i1,j1,i2,j2, pt = periodicity, lt = lattype, pvecs = pvecs(top), l = glength(layer(top)))
end
export getDistFunc


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