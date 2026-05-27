# This page is AI generated.

export SquareTopology

struct SquareTopology{U,DIMS,P <: AbstractFloat} <: AbstractLayerTopology{U,DIMS}
    size::NTuple{DIMS,Int}
    lattice_constants::MVector{DIMS,P}
    origin::SVector{DIMS,P}
end

"""
    SquareTopology(size = tuple(); lattice_constants, origin, periodic)

Create an orthogonal lattice topology with one lattice constant per dimension.
Coordinates are wrapped on periodic axes and rejected outside non-periodic axes.
"""
function SquareTopology(
    size::S = tuple();
    lattice_constants = tuple(fill(1.0, length(size))...),
    origin = tuple(fill(0.0, length(size))...),
    periodic::Union{Bool,<:Tuple,Nothing} = true,
) where {S<:Tuple}
    @assert periodic == true ? !isempty(size) : true "Size must be given if periodic is true"
    @assert length(lattice_constants) == length(size) "lattice_constants must be same length as size"
    @assert length(origin) == length(size) "origin must be same length as size"

    # Encode periodicity in the topology type so downstream dispatch can specialize.
    U = if periodic isa Bool
        periodic ? Periodic : NonPeriodic
    elseif isnothing(periodic)
        Periodic
    else
        PartPeriodic(periodic...)
    end

    DIMS = length(size)
    P = DIMS == 0 ? Float64 : promote_type(typeof(float(lattice_constants[1])), typeof(float(origin[1])))
    return SquareTopology{U,DIMS,P}(
        size,
        MVector{DIMS,P}(tuple(P.(lattice_constants)...)),
        SVector{DIMS,P}(tuple(P.(origin)...)),
    )
end

"""
    lattice_constants(top)

Return the mutable per-axis lattice constants used for square-lattice distances.
"""
@inline lattice_constants(top::T) where {T<:SquareTopology} = top.lattice_constants

"""
    origin(top)

Return the world-space origin of a square topology.
"""
@inline origin(top::T) where {T<:SquareTopology} = top.origin

"""
    setdist!(top, lattice_constants)

Update the per-axis lattice constants in-place and return `top`.
"""
function setdist!(lt::T, lattice_constants::NTuple{DIMS}) where {Periodicity,DIMS,P,T<:SquareTopology{Periodicity,DIMS,P}}
    lt.lattice_constants .= lattice_constants
    return lt
end

"""
    top(delta)

Return the topology-adjusted coordinate delta for square boundary conditions.
"""
function (lt::SquareTopology{U,DIMS,P})(d::D) where {U,DIMS,P,D<:DeltaCoordinate}
    U <: NonPeriodic && return d
    @assert length(d) == length(size(lt))

    wrapped = ntuple(Val(length(d.deltas))) do i
        di = d.deltas[i]
        if U <: Periodic || periodic(U(), Val(i))
            abs(di) > size(lt, i) / 2 && (di -= sign(di) * size(lt, i))
        end
        di
    end
    return DeltaCoordinate(wrapped...)
end

"""
    in(coord, top)

Return whether `coord` lies inside a non-periodic square topology.
"""
function Base.in(coord, lt::SquareTopology{NonPeriodic})
    return all(1 .<= coord .<= size(lt))
end

"""
    in(coord, top)

Return true for a fully periodic square topology because all coordinates wrap.
"""
@inline Base.in(coord, lt::T) where {T<:SquareTopology{Periodic}} = true

"""
    in(coord, top)

Return whether `coord` lies inside all non-periodic axes of a partly periodic
square topology.
"""
function Base.in(coord, lt::T) where {P<:PartPeriodic,T<:SquareTopology{P}}
    _isin = true
    for (i, isperiodic) in enumerate(whichperiodic(lt))
        !_isin && break
        isperiodic && continue
        _isin &= (1 <= coord[i] <= size(lt, i))
    end
    return _isin
end
