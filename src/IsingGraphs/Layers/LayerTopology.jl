using LinearAlgebra, StaticArrays

# TwoVec = SVector{2,Float32}
# TwoVec = Point2f

abstract type LatticeType end
abstract type LayerTopology{U <: PeriodicityType, Dim} end

struct Square <: LatticeType end
struct Rectangular <: LatticeType end
struct Oblique <: LatticeType end
struct Hexagonal <: LatticeType end
struct Rhombic <: LatticeType end
struct AnyLattice <: LatticeType end
export LatticeType, Square, Rectangular, Oblique, Hexagonal, Rhombic, AnyLattice

struct SquareTopology{U,DIMS} <: LayerTopology{U, DIMS}
    size::NTuple{DIMS,Int32}
    function SquareTopology(size; periodic::Bool = true)
        U = periodic ? Periodic : NonPeriodic
        DIMS = length(size)
        new{U, DIMS}(size)
    end
end
mutable struct LatticeTopology{T <: LatticeType, U <: PeriodicityType, Dim} <: LayerTopology{U, Dim} 
    # layer::Union{Nothing, AbstractIsingGraph}
    pvecs::NTuple{Dim, SVector{Dim, Float32}}
    covecs::NTuple{Dim, SVector{Dim, Float32}}
    size::NTuple{Dim, Int32}

    function LatticeTopology(_size::Tuple, vec1::Union{Nothing,AbstractArray} = nothing, vec2::Union{Nothing,AbstractArray} = nothing, vec3::Union{Nothing,AbstractArray} = nothing; periodic::Union{Nothing, Bool} = nothing)
        D = DIMS(layer)
        
        if D == 2 
            # Assert either none given or both
            @assert (isnothing(vec1) && isnothing(vec2)) || (!isnothing(vec1) && !isnothing(vec2))
             #calculation of covectors

            # If all nothing just set to square
            if isnothing(vec1) && isnothing(vec2)
                vecs = SVector.((1f0,0f0), (0f0,1f0))
                covs = SVector.((1f0,0f0), (0f0,1f0))

                return new{Square, U, 2}(vecs, covs, _size)
            end

            y1 = 1/(vec1[1]-(vec1[2]*vec2[1]/vec2[2]))
            x1 = 1/(vec1[2]-(vec1[1]*vec2[2]/vec2[1]))
            y2 = 1/(vec2[1]-(vec2[2]*vec1[1]/vec1[2]))
            x2 = 1/(vec2[2]-(vec2[1]*vec1[2]/vec1[1]))
            
            cov1 = SVector((y1, x1))
            cov2 = SVector(y2, x2)
            
            if !isnothing(periodic)
                ptype = periodic ? Periodic : NonPeriodic
            else
                ptype = Periodic
            end

            if vec1 == [1,0] && vec2 == [0,1]
                lattice_type = Square
            end
                
        
            # return new{lattice_type, ptype, typeof(layer)}(layer, (vec1, vec2), (cov1, cov2))
            return new{lattice_type, ptype, D}( Float32.(vec1, vec2), Float32.(cov1, cov2), _size)
        elseif DIMS(layer) == 3
            # Assert either none given or all
            @assert (isnothing(vec1) && isnothing(vec2) && isnothing(vec3)) || (!isnothing(vec1) && !isnothing(vec2) && !isnothing(vec3))

            if isnothing(vec1) && isnothing(vec2) && isnothing(vec3)
                vecs = SVector.((1f0,0f0,0f0), (0f0,1f0,0f0), (0f0,0f0,1f0))
                covs = SVector.((1f0,0f0,0f0), (0f0,1f0,0f0), (0f0,0f0,1f0))

                return new{Square, U, 3}(vecs, covs, _size)
            end

            #calculation of covectors
            # det(A) = a(ei - fh) - b(di - fg) + c(dh - eg)
            det = vec1[1]*(vec2[2]*vec3[3] - vec2[3]*vec3[2]) - vec2[1]*(vec1[2]*vec3[3] - vec3[2]*vec1[3]) + vec3[1]*(vec1[2]*vec2[3] - vec2[2]*vec1[3])
            entries1 = ((vec2[2]*vec3[3] - vec3[2]*vec2[3])/det, -(vec1[2]*vec3[3]-vec3[2]*vec1[3])/det, (vec1[2]*vec2[3]-vec2[2]*vec1[3])/det) 
            entries2 = (-(vec2[1]*vec3[3] - vec3[1]*vec2[3])/det, (vec1[1]*vec3[3]-vec3[1]*vec1[3])/det, -(vec1[1]*vec2[3]-vec2[1]*vec1[3])/det)
            entries3 = ((vec2[1]*vec3[2] - vec3[1]*vec2[2])/det, -(vec1[1]*vec3[2]-vec3[1]*vec1[2])/det, (vec1[1]*vec2[2]-vec2[1]*vec1[2])/det)
            cov1 = SVector(entries1)
            cov2 = SVector(entries2)
            cov3 = SVector(entries3)

            if !isnothing(periodic)
                ptype = periodic ? Periodic : NonPeriodic
            else
                ptype = Periodic
            end

            #TODO Add support for other lattices
            lattice_type = AnyLattice
            
            return new{AnyLattice, ptype, D}( Float32.(vec1, vec2, vec3), Float32.(cov1, cov2, cov3), _size)
        end
       
    end
end

# mutable struct LayerTopology{dims, T <: PeriodicityType, U <: LatticeType, LayerType <: IsingLayer}
#     layer::Union{Nothing, LayerType}
#     pvecs::NTuple{dims,Point{dims, Float32}}
#     covecs::NTuple{dims,Point{dims, Float32}}
    
#     function LayerTopology(layer, vec1::AbstractArray, vec2::AbstractArray; periodic::Union{Nothing, Bool} = nothing)
#         #calculation of covectors
#         y1 = 1/(vec1[1]-(vec1[2]*vec2[1]/vec2[2]))
#         x1 = 1/(vec1[2]-(vec1[1]*vec2[2]/vec2[1]))
#         y2 = 1/(vec2[1]-(vec2[2]*vec1[1]/vec1[2]))
#         x2 = 1/(vec2[2]-(vec2[1]*vec1[2]/vec1[1]))
        
#         cov1 = TwoVec(y1, x1)
#         cov2 = TwoVec(y2, x2)
        
#         if !isnothing(periodic)
#             ptype = periodic ? Periodic : NonPeriodic
#         else
#             ptype = Periodic
#         end

#         lattice_type = AnyLattice

#         if vec1 == [1,0] && vec2 == [0,1]
#             lattice_type = Square
#         end
            
    
#         return new{ptype, lattice_type, typeof(layer)}(layer, (vec1, vec2), (cov1, cov2))
#     end
# end


LatticeTopology(tp::LayerTopology; periodic::Bool) = LayerTopology(tp.layer, tp.pvecs[1], tp.pvecs[2]; periodic)
LatticeTopology(tp::LayerTopology, pt::Type{<:PeriodicityType}) = LayerTopology(tp.layer, tp.pvecs[1], tp.pvecs[2], periodic = pt == Periodic ? true : false)

# changePeriodicity = 
export LayerTopology

periodic(top::LayerTopology{T,U}) where {T,U} = T
latticetype(top::LayerTopology{T,U}) where {T,U} = U
export periodic, latticetype
@setterGetter LatticeTopology size
size(top::LayerTopology) = top.size
size(top, i) = top.size[i]

function pos(i,j, pvecs)
    return i*pvecs[1] + j*pvecs[2]
end

pos(idx, top) = pos(idxToCoord(idx, glength(layer(top))), top)
export pos

###### Square Lattice
##########
function dist(top::LayerTopology, coords::T...) where T
    @inline sqrt(dist2(top, coords...))
end

function dist(top::SquareTopology{P, 2}, i1, j1, i2, j2) where P
    if P == Periodic 
        i1 = i1 > size(top,1) ? i1 - size(top,1) : i1
        i2 = i2 > size(top,1) ? i2 - size(top,1) : i2
        j1 = j1 > size(top,2) ? j1 - size(top,2) : j1
        j2 = j2 > size(top,2) ? j2 - size(top,2) : j2
    end

    return abs(i1 - i2) + abs(j1 - j2)
end

function dist2(top::SquareTopology{P, 2}, i1, j1, i2, j2) where P
    if P == Periodic 
        i1 = i1 > size(top,1) ? i1 - size(top,1) : i1
        i2 = i2 > size(top,1) ? i2 - size(top,1) : i2
        j1 = j1 > size(top,2) ? j1 - size(top,2) : j1
        j2 = j2 > size(top,2) ? j2 - size(top,2) : j2
    end

    return (i1 - i2)^2 + (j1 - j2)^2
end


function dist2(top::SquareTopology{P,3}, (i1,j1,k1)::Tuple,(i2,j2,k2)::Tuple) where P
    if P == Periodic
        i1 = i1 > size(top,1) ? i1 - size(top,1) : i1
        j1 = j1 > size(top,2) ? j1 - size(top,2) : j1
        k1 = k1 > size(top,3) ? k1 - size(top,3) : k1

        i2 = i2 > size(top,1) ? i2 - size(top,1) : i2
        j2 = j2 > size(top,2) ? j2 - size(top,2) : j2
        k2 = k2 > size(top,3) ? k2 - size(top,3) : k2
    end

    return (i1 - i2)^2 + (j1 - j2)^2 + (k1 - k2)^2
end

# If only two given must be indexes of the same layer (or in 1D case idx = i)
function dist2(top::LatticeTopology, idx1::Integer, idx2::Integer)
    coords1 = idxToCoord(Int32(idx1), size(top))
    coords2 = idxToCoord(Int32(idx2), size(top))

    return @inline dist2(top, coords1..., coords2...)
end

function dist2(top::LatticeTopology, coords1::Tuple, coords2::Tuple)
    return @inline dist2(top, coords1..., coords2...)
end

function dist(top::LatticeTopology, idx1::Integer, idx2::Integer)
    return sqrt(dist2(top, idx1, idx2))
end

function dist(top::LatticeTopology, coords1::Tuple, coords2::Tuple)
    return sqrt(dist2(top, coords1, coords2))
end



# @inline function dist2(i1, j1, i2, j2, pt::Type{NonPeriodic}, lt::LT; pvecs, l = 1, w = 1) where LT <: Union{Type{Square}, Type{AnyLattice}}
#     return sum((pos(i1,j1, pvecs) - pos(i2,j2, pvecs)).^2)
# end

# @inline function dist2(i1, j1, i2, j2, pt::Type{Periodic}, lt::LT; pvecs, l, w) where LT <: Union{Type{Square}, Type{AnyLattice}}
#     i1 = i1 > l ? i1 - l : i1
#     i2 = i2 > l ? i2 - l : i2
#     j1 = j1 > w ? j1 - w : j1
#     j2 = j2 > w ? j2 - w : j2

#     dists =  pos(i2,j2, pvecs) - pos(i1,j1, pvecs)
    
#     dy = abs(dists[1]) > l/2 ? dists[1] - sign(dists[1]) * l : dists[1]
#     dx = abs(dists[2]) > w/2 ? dists[2] - sign(dists[2]) * w : dists[2]
   
#     return dx^2+dy^2
# end

# dist2(i1, j1, i2, j2, top::LT) where {LT <: LayerTopology} = dist2(i1, j1, i2, j2, periodic(top), type(top), pvecs = pvecs(top), l = glength(layer(top)), w = gwidth(layer(top)))

# function dist(i1, j1, i2, j2, top::LayerTopology{PT,LT, LayerT}) where {PT, LT, LayerT}
#     pvecs_val = pvecs(top)
#     l::Int32 = glength(layer(top))
#     w::Int32 = gwidth(layer(top))

#     return sqrt(dist2(i1, j1, i2, j2, PT, LT, pvecs = pvecs_val, l = l, w = w))
# end

# function dist(i1, j1, k1, i2, j2, k2, top::LayerTopology{PT,LT, LayerT}) where {PT, LT, LayerT}
#     pvecs_val = pvecs(top)
#     l::Int32 = glength(layer(top))
#     w::Int32 = gwidth(layer(top))
#     h::Int32 = size(top,3)

#     return sqrt(dist2(i1, j1, i2, j2, PT, LT, pvecs = pvecs_val, l = l, w = w) + dist2(0,0,k1,k2, PT, LT, pvecs = pvecs_val, l = l, w = w))
# end

# dist(idx1::Integer,idx2::Integer, top::LayerTopology) = dist(idxToCoord(idx1, glength(layer(top)))..., idxToCoord(idx2, glength(layer(top)))..., top)

export dist2, dist

function getDistFunc(top::LT) where {LT <: LayerTopology}
    return (i1,j1,i2,j2) -> dist(top, i1,j1,i2,j2)
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

function dx(lt::SquareTopology{NonPeriodic,DIMS}, coords1::Tuple, coords2::Tuple) where DIMS
    return abs(coords2[1] - coords1[1])
end

function dx(lt::SquareTopology{Periodic,DIMS}, coords1::Tuple, coords2::Tuple) where DIMS
    di = coords2[1] - coords1[1]
    if di > size(lt,1)/2
        di -= size(lt,1)
    end
    return di
end

function dy(lt::SquareTopology{NonPeriodic,DIMS}, coords1::Tuple, coords2::Tuple) where DIMS
    return abs(coords2[1] - coords1[1])
end

function dy(lt::SquareTopology{Periodic,DIMS}, coords1::Tuple, coords2::Tuple) where DIMS
    dj = coords2[2] - coords1[2]
    if dj > size(lt,2)/2
        dj -= size(lt,2)
    end
    return dj
end

function dz(lt::SquareTopology{NonPeriodic,3}, coords1::Tuple, coords2::Tuple)
    return abs(coords2[3] - coords1[3])
end

function dz(lt::SquareTopology{Periodic,3}, coords1::Tuple, coords2::Tuple)
    dk = coords2[3] - coords1[3]
    if dk > size(lt,3)/2
        dk -= size(lt,3)
    end
    return dk
end

dxdy(lt::SquareTopology{P,2}, coords1::Tuple, coords2::Tuple) where P = (dx(lt, coords1, coords2), dy(lt, coords1, coords2))

dxdydz(lt::SquareTopology{P,3}, coords1::Tuple, coords2::Tuple) where P = (dx(lt, coords1, coords2), dy(lt, coords1, coords2), dz(lt, coords1, coords2))


# @inline function dx(i1, j1, i2, j2, layer, lt::LayerTopology{Periodic,A,B} = top(layer)) where {A,B}
#     di = i2 - i1
#     if di > glength(layer)/Int32(2)
#         di -= glength(layer)
#     end
#     # di = di > glength(layer)/2 ? di - glength(layer) : di
#     dj = j2 - j1
#     # dj = dj > gwidth(layer)/2 ? dj - gwidth(layer) : dj
#     if dj > gwidth(layer)/Int32(2)
#         dj -= gwidth(layer)
#     end

#     return  di*pvecs(lt)[1][1] + dj*pvecs(lt)[2][1]
# end

# @inline function dy(i1, j1, i2, j2, layer, lt::LayerTopology{Periodic,A,B} = top(layer)) where {A,B}
#     dj = j2 - j1
#     dj = dj > gwidth(layer)/2 ? dj - gwidth(layer) : dj
#     di = i2 - i1
#     di = di > glength(layer)/2 ? di - glength(layer) : di
#     return di*pvecs(lt)[1][2] + dj*pvecs(lt)[2][2]
# end

# @inline function dxdy(i1, j1, i2, j2, layer, lt::LayerTopology{Periodic,A,B} = top(layer)) where {A,B}
#     return (dx(i1,j1,i2,j2, layer, lt), dy(i1,j1,i2,j2, layer, lt))
# end

# @inline dx(idx1::Integer,idx2::Integer, layer) = dx(idxToCoord(idx1, layer)..., idxToCoord(idx2, layer)..., layer)
# @inline dy(idx1::Integer,idx2::Integer, layer) = dy(idxToCoord(idx1, layer)..., idxToCoord(idx2, layer)..., layer)

# @inline dxdy(idx1::Integer,idx2::Integer, layer) = (dx(idx1,idx2, layer), dy(idx1,idx2, layer))

export dy, dx, dxdy