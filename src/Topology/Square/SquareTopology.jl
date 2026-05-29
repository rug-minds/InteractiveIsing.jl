# This page is AI generated.

export SquareTopology

struct SquareTopology{U,DIMS,P <: AbstractFloat} <: AbstractLayerTopology{U,DIMS}
    size::NTuple{DIMS,Int}
    lattice_constants::MVector{DIMS,P}
    origin::SVector{DIMS,P}
end

"""
    SquareTopology([size]; lattice_constants, origin, periodic)

Create an orthogonal lattice topology with one lattice constant per dimension.
Coordinates are wrapped on periodic axes and rejected outside non-periodic axes.
Omitting `size` creates a topology template when the dimension can be inferred
from `lattice_constants` or `origin`.
"""
function SquareTopology(
    size::S = tuple();
    lattice_constants = nothing,
    origin = nothing,
    periodic::Union{Bool,<:Tuple,Nothing} = true,
) where {S<:Tuple}
    DIMS = isempty(size) ? _square_template_dim(lattice_constants, origin) : length(size)
    size_int = isempty(size) ? ntuple(_ -> 0, Val(DIMS)) : ntuple(i -> Int(size[i]), Val(DIMS))
    constants = isnothing(lattice_constants) ? tuple(fill(1.0, DIMS)...) : tuple(lattice_constants...)
    origin_tuple = isnothing(origin) ? tuple(fill(0.0, DIMS)...) : tuple(origin...)

    @assert length(constants) == DIMS "lattice_constants must be same length as size"
    @assert length(origin_tuple) == DIMS "origin must be same length as size"

    # Encode periodicity in the topology type so downstream dispatch can specialize.
    U = if periodic isa Bool
        periodic ? Periodic : NonPeriodic
    elseif isnothing(periodic)
        Periodic
    else
        PartPeriodic(periodic...)
    end

    P = promote_type(typeof(float(constants[1])), typeof(float(origin_tuple[1])))
    return SquareTopology{U,DIMS,P}(
        size_int,
        MVector{DIMS,P}(tuple(P.(constants)...)),
        SVector{DIMS,P}(tuple(P.(origin_tuple)...)),
    )
end

"""
    _square_template_dim(lattice_constants, origin)

Infer the dimension of an unsized square topology template.
"""
function _square_template_dim(lattice_constants, origin)
    if !isnothing(lattice_constants)
        D = length(lattice_constants)
    elseif !isnothing(origin)
        D = length(origin)
    else
        throw(ArgumentError("Unsized SquareTopology needs lattice_constants or origin to infer dimension."))
    end
    D > 0 || throw(ArgumentError("SquareTopology dimension must be positive."))
    return D
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
    sizeto(top, size)

Return a square topology sized to a layer. Unsized templates have zero extents;
fully sized topologies must already match.
"""
function sizeto(top::T, layer_size::NTuple{DIMS,<:Integer}) where {Periodicity,DIMS,P,T<:SquareTopology{Periodicity,DIMS,P}}
    requested_size = ntuple(i -> Int(layer_size[i]), Val(DIMS))
    current_size = size(top)
    if current_size == requested_size
        return top
    elseif all(iszero, current_size)
        return SquareTopology{Periodicity,DIMS,P}(
            requested_size,
            MVector{DIMS,P}(Tuple(top.lattice_constants)),
            top.origin,
        )
    else
        throw(ArgumentError("Explicit topology size $(current_size) does not match layer size $(requested_size)."))
    end
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
