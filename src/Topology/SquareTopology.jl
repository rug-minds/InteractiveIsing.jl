struct SquareTopology{U,DIMS, P <: AbstractFloat} <: AbstractLayerTopology{U, DIMS}
    size::NTuple{DIMS,Int}
    ds::MVector{DIMS, P}
end


function SquareTopology(size = tuple(); ds = tuple(fill(1.0, length(size))...), periodic::Union{Bool, <:Tuple, Nothing} = true)
        U = nothing

        @assert periodic == true ? !isempty(size) : true "Size must be given if periodic is true"
        @assert length(ds) == length(size) "ds must be same length as size" 

        if periodic isa Bool
            U = periodic ? Periodic : NonPeriodic
        elseif isnothing(periodic)
            U = Periodic
        else
            U = PartPeriodic(periodic...) 
        end
        DIMS = length(size)
        SquareTopology{U, DIMS, eltype(ds)}(size, tuple(ds...))
end

"""
Get the lattice constants of a square topology.
"""
lattice_constants(top::SquareTopology) = top.ds

setdist!(lt::SquareTopology{U,DIMS,P}, ds::NTuple{DIMS,P}) where {U,DIMS,P} = begin lt.ds .= ds; lt end

"""
Get the distance from a deltacoordinate, applying periodic boundary conditions if necessary.
"""
function (lt::SquareTopology{Periodic})(d::DeltaCoordinate)
    @assert length(d) == length(size(lt))
    function get_taurus_dist(di, size_i)
        di = abs(di)
        if di > size_i/2
            di -= size_i
        end
        return di
    end
    DeltaCoordinate(ntuple(i -> get_taurus_dist(d.deltas[i], size(lt,i)), Val(length(d.deltas)))...)
end

function (lt::SquareTopology{NonPeriodic})(d::DeltaCoordinate)
    return d
end

function Base.in(coord, lt::SquareTopology{NonPeriodic})
    all(1 .<= coord .<= size(lt))
end

Base.in(coord, lt::SquareTopology{Periodic}) = true

function Base.in(coord, lt::SquareTopology{P}) where {P <: PartPeriodic}
    _isin = true
    for (i,isperiodic) in enumerate(whichperiodic(lt))
        if !_isin
            break
        end
        if isperiodic
            continue
        end
        _isin &= (1 <= coord[i] <= size(lt,i))
    end
    _isin
end