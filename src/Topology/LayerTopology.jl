
export LatticeType, Square, Rectangular, Oblique, Hexagonal, Rhombic, AnyLattice, AbstractLayerTopology, periodic, latticetype

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
    pvecs::NTuple{Dim, SVector{Dim, Precision}}
    covecs::NTuple{Dim, SVector{Dim, Precision}}
    size::NTuple{Dim, Int32}
    origin::SVector{Dim, Precision}

    # This function uses SMatrix Inverse to calculate the covectors
    function LatticeTop(_size::NTuple{D}, vecs::NTuple{D, <:AbstractArray}; periodic = nothing, precision = Float32, origin = SVector(ntuple(i->precision(0), D))) where D
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

        return new{Square, U, D, precision}(vecs, covs, _size, origin; periodic)
    end

    function LatticeTopology(_size::Tuple, 
        vec1::Union{Nothing,AbstractArray} = nothing, 
        vec2::Union{Nothing,AbstractArray} = nothing, 
        vec3::Union{Nothing,AbstractArray} = nothing; 
        periodic::Union{Nothing, Bool, Tuple} = nothing, 
        origin::Union{Nothing, AbstractArray} = nothing)

        D = length(_size)
        
        ##### Calculate the covectors
        if D == 2 
            # Assert either none given or both
            @assert (isnothing(vec1) && isnothing(vec2)) || (!isnothing(vec1) && !isnothing(vec2))

            if !isnothing(origin)
                origin = SVector{D, eltype(origin)}(origin...)
            else
                origin = SVector{D, Float32}(ntuple(i->0f0, D)...)
            end

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

@Setter!Getter LatticeTopology size
Base.size(top::AbstractLayerTopology) = top.size
Base.size(top, i) = top.size[i]


LatticeTopology(tp::AbstractLayerTopology; periodic::Bool) = AbstractLayerTopology(tp.layer, tp.pvecs[1], tp.pvecs[2]; periodic)
LatticeTopology(tp::AbstractLayerTopology, pt::Type{<:PeriodicityType}) = AbstractLayerTopology(tp.layer, tp.pvecs[1], tp.pvecs[2], periodic = pt == Periodic ? true : false)

# changePeriodicity = 

periodic(top::AbstractLayerTopology{T,U}) where {T,U} = T
latticetype(top::AbstractLayerTopology{T,U}) where {T,U} = U

"""
Calculate the position based on indices and primitive vectors.
"""
function coord(pvecs, i,j)
    return i*pvecs[1] + j*pvecs[2]
end

function coordinate_generator(top::AbstractLayerTopology)
    
end