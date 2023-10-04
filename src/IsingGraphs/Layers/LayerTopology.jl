using LinearAlgebra

# TwoVec = SVector{2,Float32}
TwoVec = Point2f

abstract type LatticeType end
struct Square <: LatticeType end
struct Rectangular <: LatticeType end
struct Oblique <: LatticeType end
struct Hexagonal <: LatticeType end
struct Rhombic <: LatticeType end
struct AnyLattice <: LatticeType end
export LatticeType, Square, Rectangular, Oblique, Hexagonal, Rhombic, AnyLattice

mutable struct LayerTopology{T <: PeriodicityType, U <: LatticeType, LayerType <: IsingLayer}
    layer::Union{Nothing, LayerType}
    pvecs::Tuple{TwoVec, TwoVec}
    covecs::Tuple{TwoVec, TwoVec}

    function LayerTopology(layer, vec1::AbstractArray, vec2::AbstractArray; periodic::Union{Nothing, Bool} = nothing)
        #calculation of covectors
        y1 = 1/(vec1[1]-(vec1[2]*vec2[1]/vec2[2]))
        x1 = 1/(vec1[2]-(vec1[1]*vec2[2]/vec2[1]))
        y2 = 1/(vec2[1]-(vec2[2]*vec1[1]/vec1[2]))
        x2 = 1/(vec2[2]-(vec2[1]*vec1[2]/vec1[1]))
        
        cov1 = TwoVec(y1, x1)
        cov2 = TwoVec(y2, x2)
        
        if !isnothing(periodic)
            ptype = periodic ? Periodic : NonPeriodic
        else
            ptype = Periodic
        end

        lattice_type = AnyLattice

        if vec1 == [1,0] && vec2 == [0,1]
            lattice_type = Square
        end
            
    
        return new{ptype, lattice_type, typeof(layer)}(layer, (vec1, vec2), (cov1, cov2))
    end
end

LayerTopology(tp::LayerTopology; periodic::Bool) = LayerTopology(tp.layer, tp.pvecs[1], tp.pvecs[2]; periodic)
LayerTopology(tp::LayerTopology, pt::Type{<:PeriodicityType}) = LayerTopology(tp.layer, tp.pvecs[1], tp.pvecs[2], periodic = pt == Periodic ? true : false)

# changePeriodicity = 
export LayerTopology

periodic(top::LayerTopology{T,U,L}) where {T,U,L} = T
latticetype(top::LayerTopology{T,U,L}) where {T,U,L} = U
export periodic, latticetype
@setterGetter LayerTopology

function pos(i,j, pvecs)
    return i*pvecs[1] + j*pvecs[2]
end

pos(idx, top) = pos(idxToCoord(idx, glength(layer(top))), top)
export pos

@inline function dist2(i1, j1, i2, j2, pt::Type{NonPeriodic}, lt::LT; pvecs, l = 1, w = 1) where LT <: Union{Type{Square}, Type{AnyLattice}}
    return sum((pos(i1,j1, pvecs) - pos(i2,j2, pvecs)).^2)
end

@inline function dist2(i1, j1, i2, j2, pt::Type{Periodic}, lt::LT; pvecs, l, w) where LT <: Union{Type{Square}, Type{AnyLattice}}
    i1 = i1 > l ? i1 - l : i1
    i2 = i2 > l ? i2 - l : i2
    j1 = j1 > w ? j1 - w : j1
    j2 = j2 > w ? j2 - w : j2

    dists =  pos(i2,j2, pvecs) - pos(i1,j1, pvecs)
    
    dy = abs(dists[1]) > l/2 ? dists[1] - sign(dists[1]) * l : dists[1]
    dx = abs(dists[2]) > w/2 ? dists[2] - sign(dists[2]) * w : dists[2]
   
    return dx^2+dy^2
end

dist2(i1, j1, i2, j2, top::LT) where {LT <: LayerTopology} = dist2(i1, j1, i2, j2, periodic(top), type(top), pvecs = pvecs(top), l = glength(layer(top)), w = gwidth(layer(top)))

function dist(i1, j1, i2, j2, top::LayerTopology{PT,LT, LayerT}) where {PT, LT, LayerT}
    pvecs_val = pvecs(top)
    l::Int32 = glength(layer(top))
    w::Int32 = gwidth(layer(top))

    return sqrt(dist2(i1, j1, i2, j2, PT, LT, pvecs = pvecs_val, l = l, w = w))
end

dist(idx1::Integer,idx2::Integer, top::LayerTopology) = dist(idxToCoord(idx1, glength(layer(top)))..., idxToCoord(idx2, glength(layer(top)))..., top)

export dist2, dist

function getDistFunc(top::LT) where {LT <: LayerTopology}
    return (i1,j1,i2,j2) -> dist(i1,j1,i2,j2, top)
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

@inline function dx(i1, j1, i2, j2, layer, lt::LayerTopology{Periodic,A,B} = top(layer)) where {A,B}
    di = i2 - i1
    if di > glength(layer)/Int32(2)
        di -= glength(layer)
    end
    # di = di > glength(layer)/2 ? di - glength(layer) : di
    dj = j2 - j1
    # dj = dj > gwidth(layer)/2 ? dj - gwidth(layer) : dj
    if dj > gwidth(layer)/Int32(2)
        dj -= gwidth(layer)
    end

    return  di*pvecs(lt)[1][1] + dj*pvecs(lt)[2][1]
end

@inline function dy(i1, j1, i2, j2, layer, lt::LayerTopology{Periodic,A,B} = top(layer)) where {A,B}
    dj = j2 - j1
    dj = dj > gwidth(layer)/2 ? dj - gwidth(layer) : dj
    di = i2 - i1
    di = di > glength(layer)/2 ? di - glength(layer) : di
    return di*pvecs(lt)[1][2] + dj*pvecs(lt)[2][2]
end

@inline function dxdy(i1, j1, i2, j2, layer, lt::LayerTopology{Periodic,A,B} = top(layer)) where {A,B}
    return (dx(i1,j1,i2,j2, layer, lt), dy(i1,j1,i2,j2, layer, lt))
end

@inline dx(idx1::Integer,idx2::Integer, layer) = dx(idxToCoord(idx1, layer)..., idxToCoord(idx2, layer)..., layer)
@inline dy(idx1::Integer,idx2::Integer, layer) = dy(idxToCoord(idx1, layer)..., idxToCoord(idx2, layer)..., layer)

@inline dxdy(idx1::Integer,idx2::Integer, layer) = (dx(idx1,idx2, layer), dy(idx1,idx2, layer))

export dy, dx, dxdy