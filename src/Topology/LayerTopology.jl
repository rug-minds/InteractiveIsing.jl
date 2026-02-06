
export LatticeType, Square, Rectangular, Oblique, Hexagonal, Rhombic, AnyLattice




struct Square <: LatticeType end
struct Rectangular <: LatticeType end
struct Oblique <: LatticeType end
struct Hexagonal <: LatticeType end
struct Rhombic <: LatticeType end
struct AnyLattice <: LatticeType end


"""
General non-square layer topology
"""
mutable struct LatticeTopology{T <: LatticeType, U, Dim, Precision} <: AbstractLayerTopology{U, Dim} 
    # layer::Union{Nothing, AbstractIsingGraph}
    pvecs::NTuple{Dim, SVector{Dim, Precision}}
    covecs::NTuple{Dim, SVector{Dim, Precision}}
    size::NTuple{Dim, Int32}

    # This function uses SMatrix Inverse to calculate the covectors
    function LatticeTop(_size::NTuple{D}, vecs::NTuple{D, <:AbstractArray}; periodic = nothing, precision = Float32) where D
        # D = length(_size)
        vecs = Vector{SVector{D, precision}}(undef, D)
        for i in 1:D
            if i <= length(vecs)
                vecs[i] = SVector{D, precision}(vecs[i]...)
            else
                vecs[i] = SVector{D, precision}([i == j ? one(precision) : zero(precision) for j in 1:D])
            end
        end
        vecs = tuple(vecs...)
        m = SMatrix{D, D, precision}(vecs...)
        covs = tuple(eachcol(inv(m))...)  # Calculate the covectors using the inverse of the matrix

        return new{Square, U, D, precision}(vecs, covs, _size)
    end

    function LatticeTopology(_size::Tuple, vec1::Union{Nothing,AbstractArray} = nothing, vec2::Union{Nothing,AbstractArray} = nothing, vec3::Union{Nothing,AbstractArray} = nothing; periodic::Union{Nothing, Bool, Tuple} = nothing)
        D = length(_size)
        
        ##### Calculate the covectors
        if D == 2 
            # Assert either none given or both
            @assert (isnothing(vec1) && isnothing(vec2)) || (!isnothing(vec1) && !isnothing(vec2))
             #calculation of covectors

            # If all nothing just set to square
            if isnothing(vec1) && isnothing(vec2)
                vecs = SVector.((1f0,0f0), (0f0,1f0))
                covs = SVector.((1f0,0f0), (0f0,1f0))

                return new{Square, U, 2, Float32}(vecs, covs, _size)
            end

            y1 = 1/(vec1[1]-(vec1[2]*vec2[1]/vec2[2]))
            x1 = 1/(vec1[2]-(vec1[1]*vec2[2]/vec2[1]))
            y2 = 1/(vec2[1]-(vec2[2]*vec1[1]/vec1[2]))
            x2 = 1/(vec2[2]-(vec2[1]*vec1[2]/vec1[1]))
            
            cov1 = SVector((y1, x1))
            cov2 = SVector(y2, x2)
            
            if !isnothing(periodic)
                if periodic isa Bool
                    ptype = periodic ? Periodic : NonPeriodic
                elseif periodic isa Tuple
                    ptype = PartPeriodic(periodic...)
                end
            else
                ptype = Periodic
            end

            if vec1 == [1,0] && vec2 == [0,1]
                lattice_type = Square
            end
                
        
            # return new{lattice_type, ptype, typeof(layer)}(layer, (vec1, vec2), (cov1, cov2))
            return new{lattice_type, ptype, D, Float32}( Float32.(vec1, vec2), Float32.(cov1, cov2), _size)
        elseif DIMS(layer) == 3
            # Assert either none given or all
            @assert (isnothing(vec1) && isnothing(vec2) && isnothing(vec3)) || (!isnothing(vec1) && !isnothing(vec2) && !isnothing(vec3))

            if isnothing(vec1) && isnothing(vec2) && isnothing(vec3)
                vecs = SVector.((1f0,0f0,0f0), (0f0,1f0,0f0), (0f0,0f0,1f0))
                covs = SVector.((1f0,0f0,0f0), (0f0,1f0,0f0), (0f0,0f0,1f0))

                return new{Square, U, 3, Float32}(vecs, covs, _size)
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
            
            return new{AnyLattice, ptype, D, Float32}( Float32.(vec1, vec2, vec3), Float32.(cov1, cov2, cov3), _size)
        end
       
    end
end


LatticeTopology(tp::AbstractLayerTopology; periodic::Bool) = AbstractLayerTopology(tp.layer, tp.pvecs[1], tp.pvecs[2]; periodic)
LatticeTopology(tp::AbstractLayerTopology, pt::Type{<:PeriodicityType}) = AbstractLayerTopology(tp.layer, tp.pvecs[1], tp.pvecs[2], periodic = pt == Periodic ? true : false)

# changePeriodicity = 
export AbstractLayerTopology

periodic(top::AbstractLayerTopology{T,U}) where {T,U} = T
latticetype(top::AbstractLayerTopology{T,U}) where {T,U} = U
export periodic, latticetype
@setterGetter LatticeTopology size
Base.size(top::AbstractLayerTopology) = top.size
Base.size(top, i) = top.size[i]

function pos(i,j, pvecs)
    return i*pvecs[1] + j*pvecs[2]
end

pos(idx, top) = pos(idxToCoord(idx, glength(layer(top))), top)
export pos

###### Square Lattice
##########
# function dist(top::AbstractLayerTopology, coords::T...) where T
#     @inline sqrt(dist2(top, coords...))
# end

# # function dist(top::SquareTopology{P, 2}, i1, j1, i2, j2) where P
   
# # end

# function dist2(top::SquareTopology{P, 2}, i1, j1, i2, j2) where P
#     di = abs(i1 - i2)
#     dj = abs(j1 - j2)
#     if periodic(top, :x) 
#         if di > size(top,1)/2
#             di -= size(top,1)
#         end
#     end
#     if periodic(top, :y)
#         if dj > size(top,2)/2
#             dj -= size(top,2)
#         end
#     end

#     return di^2 + dj^2
# end

# dist2(top::SquareTopology{P, 3}, (i1,j1,k1)::Tuple,(i2,j2,k2)::Tuple) where P = dist2(top, i1,j1,k1,i2,j2,k2)

# function dist2(top::SquareTopology{P,3}, i1,j1,k1,i2,j2,k2) where P
#     di = abs(i1 - i2)
#     dj = abs(j1 - j2)
#     dk = abs(k1 - k2)
#     if periodic(top, :x)
#         if di > size(top,1)/2
#             di -= size(top,1)
#         end
#     end
#     if periodic(top, :y)
#         if dj > size(top,2)/2
#             dj -= size(top,2)
#         end
#     end
#     if periodic(top, :z)
#         if dk > size(top,3)/2
#             dk -= size(top,3)
#         end
#     end
    

#     return di^2 + dj^2 + dk^2
# end

# # If only two given must be indexes of the same layer (or in 1D case idx = i)
# function dist2(top::AbstractLayerTopology, idx1::Integer, idx2::Integer)
#     coords1 = idxToCoord(Int32(idx1), size(top))
#     coords2 = idxToCoord(Int32(idx2), size(top))

#     return @inline dist2(top, coords1..., coords2...)
# end

# function dist2(top::AbstractLayerTopology, coords1::Tuple, coords2::Tuple)
#     return @inline dist2(top, coords1..., coords2...)
# end

# function dist(top::AbstractLayerTopology, idx1::Integer, idx2::Integer)
#     return sqrt(dist2(top, idx1, idx2))
# end

# function dist(top::AbstractLayerTopology, coords1::Tuple, coords2::Tuple)
#     return sqrt(dist2(top, coords1, coords2))
# end

# export dist2, dist

# function getDistFunc(top::LT) where {LT <: AbstractLayerTopology}
#     return (i1,j1,i2,j2) -> dist(top, i1,j1,i2,j2)
# end
#   export getDistFunc


# function (lt::AbstractLayerTopology)(y,x)
#     point = TwoVec(y,x)
#     comp1 = point ⋅ covecs(lt)[1]
#     comp2 = point ⋅ covecs(lt)[2]

#     return (comp1, comp2)
# end

# function latToPoint(layer, i::Integer, j::Integer)
#     tp = top(layer)
#     zag = i ÷ 2
#     zig = i - zag
#     zagvec = TwoVec(pvecs(tp)[1][1], -pvecs(tp)[1][2])
#     return zig*pvecs(tp)[1] + zag*zagvec + j*pvecs(tp)[2]
# end
# export latToPoint

# function dx(lt, per::Val{false}, coords1::Tuple, coords2::Tuple)
#     return abs(coords2[1] - coords1[1])
# end

# function dx(lt, per::Val{true}, coords1::Tuple, coords2::Tuple)
#     di = coords2[1] - coords1[1]
#     if di > size(lt,1)/2
#         di -= size(lt,1)
#     end
#     return di
# end

# function dy(lt, per::Val{false}, coords1::Tuple, coords2::Tuple)
#     return abs(coords2[1] - coords1[1])
# end

# function dy(lt, per::Val{true}, coords1::Tuple, coords2::Tuple)
#     dj = coords2[2] - coords1[2]
#     if dj > size(lt,2)/2
#         dj -= size(lt,2)
#     end
#     return dj
# end

# function dz(lt, per::Val{false}, coords1::Tuple, coords2::Tuple)
#     return abs(coords2[3] - coords1[3])
# end

# function dz(lt, per::Val{true}, coords1::Tuple, coords2::Tuple)
#     dk = coords2[3] - coords1[3]
#     if dk > size(lt,3)/2
#         dk -= size(lt,3)
#     end
#     return dk
# end

# dxdy(lt::SquareTopology{P,2}, coords1::Tuple, coords2::Tuple) where P = (dx(lt, Val(periodic(lt,:x)), coords1, coords2), dy(lt, Val(periodic(lt,:y)), coords1, coords2))

# dxdydz(lt::SquareTopology{P,3}, coords1::Tuple, coords2::Tuple) where P = (dx(lt, Val(periodic(lt,:x)), coords1, coords2), dy(lt, Val(periodic(lt,:y)), coords1, coords2), dz(lt, Val(periodic(lt,:z)), coords1, coords2))

# export dy, dx, dxdy

# function lat_mod_or_in(::P, coord::Integer, coordsize::Integer) where P <: PeriodicityType
#     if P == Periodic
#         return latmod(coord, coordsize)
#     elseif P == NonPeriodic
#         return inlat(coord, coordsize)
#     end
# end

# function lat_mod_or_in(top::AbstractLayerTopology{P,N}, coord::NTuple{N,Int32}, size::NTuple{N,Int32}) where {P<:PeriodicityType,N}
#     return ((lat_mod_or_in(coordperiodicity(top, coord_symbs[i]), coord[i], size[i]) for i in 1:N)...,)
# end

# function coordperiodicity(top::AbstractLayerTopology{Periodic}, symb)
#     return Periodic
# end

# function coordperiodicity(top::AbstractLayerTopology{NonPeriodic}, symb)
#     return NonPeriodic
# end

# function coordperiodicity(top::AbstractLayerTopology{PartPeriodic{Parts}}, symb) where {Parts}
#     return symb in Parts ? Periodic : NonPeriodic
# end