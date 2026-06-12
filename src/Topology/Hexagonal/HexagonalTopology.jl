# This page is AI generated.

export HexagonalTopology

struct HexagonalTopology{U,S,L,O,Orientation} <: AbstractLayerTopology{U,2}
    size::S
    lattice_constants::L
    origin::O
end

"""
    HexagonalTopology([size]; lattice_constant = 1.0, lattice_constants = nothing, origin = (0.0, 0.0), orientation = :pointy, periodic = true)

Create a two-dimensional axial-coordinate hexagonal topology. The default
`:pointy` orientation uses primitive vectors `(a, 0)` and `(a/2, sqrt(3)a/2)`,
so the axial offsets `(1, 0)`, `(0, 1)`, and `(1, -1)` are nearest neighbors.
Omitting `size` creates an unsized topology template.
"""
function HexagonalTopology(
    size::S = (0, 0);
    lattice_constant = 1.0,
    lattice_constants = nothing,
    origin = (0.0, 0.0),
    orientation::Symbol = :pointy,
    periodic::Union{Bool,<:Tuple,Nothing} = true,
    physical_scales = nothing,
) where {S<:Tuple}
    @assert length(size) == 2 "HexagonalTopology currently supports only two-dimensional layers"
    @assert length(origin) == 2 "origin must have two entries for HexagonalTopology"
    @assert orientation in (:pointy, :flat) "orientation must be either :pointy or :flat"

    # Keep the public distance API compatible with SquareTopology while allowing
    # the common single-constant regular hexagonal case.
    constants = if isnothing(lattice_constants)
        (lattice_constant, lattice_constant)
    else
        @assert length(lattice_constants) == 2 "lattice_constants must have two entries for HexagonalTopology"
        tuple(lattice_constants...)
    end
    length_scale = _length_reference_scale(physical_scales, constants, origin)
    constants = _internal_length_container(constants, length_scale)
    origin = _internal_length_container(origin, length_scale)

    U = if periodic isa Bool
        periodic ? Periodic : NonPeriodic
    elseif isnothing(periodic)
        Periodic
    else
        PartPeriodic(periodic...)
    end

    P = promote_type(typeof(float(constants[1])), typeof(float(origin[1])))
    size_int = (Int(size[1]), Int(size[2]))
    constant_vec = MVector{2,P}(P(constants[1]), P(constants[2]))
    origin_vec = SVector{2,P}(P(origin[1]), P(origin[2]))
    return HexagonalTopology{U,typeof(size_int),typeof(constant_vec),typeof(origin_vec),orientation}(
        size_int,
        constant_vec,
        origin_vec,
    )
end

"""
    lattice_constants(top)

Return the mutable axial-axis lattice constants for a hexagonal topology.
"""
@inline lattice_constants(top::T) where {T<:HexagonalTopology} = top.lattice_constants

"""
    origin(top)

Return the world-space origin of a hexagonal topology.
"""
@inline origin(top::T) where {T<:HexagonalTopology} = top.origin

"""
    setdist!(top, lattice_constants)

Update the axial-axis lattice constants in-place and return `top`.
"""
function setdist!(lt::T, lattice_constants::NTuple{2}) where {Periodicity,S,L,O,Orientation,T<:HexagonalTopology{Periodicity,S,L,O,Orientation}}
    length_scale = _length_reference_scale(nothing, lattice_constants)
    lattice_constants = _internal_length_container(lattice_constants, length_scale)
    lt.lattice_constants .= lattice_constants
    return lt
end

"""
    sizeto(top, size)

Return a hexagonal topology sized to a layer. Unsized templates have zero
extents; fully sized topologies must already match.
"""
function sizeto(top::T, layer_size::NTuple{2,<:Integer}) where {Periodicity,S,L,O,Orientation,T<:HexagonalTopology{Periodicity,S,L,O,Orientation}}
    requested_size = (Int(layer_size[1]), Int(layer_size[2]))
    current_size = size(top)
    if current_size == requested_size
        return top
    elseif all(iszero, current_size)
        return HexagonalTopology{Periodicity,typeof(requested_size),L,O,Orientation}(
            requested_size,
            copy(top.lattice_constants),
            top.origin,
        )
    else
        throw(ArgumentError("Explicit topology size $(current_size) does not match layer size $(requested_size)."))
    end
end

"""
    primitive_vectors(top)

Return the two world-space primitive vectors used by a hexagonal topology.
"""
@inline function primitive_vectors(top::T) where {U,S,L,O,T<:HexagonalTopology{U,S,L,O,:pointy}}
    a1, a2 = lattice_constants(top)
    return (
        SVector(a1, zero(a1)),
        SVector(a2 / 2, sqrt(a2 * a2 * 3) / 2),
    )
end

"""
    primitive_vectors(top)

Return the two world-space primitive vectors for a flat-top hexagonal topology.
"""
@inline function primitive_vectors(top::T) where {U,S,L,O,T<:HexagonalTopology{U,S,L,O,:flat}}
    a1, a2 = lattice_constants(top)
    return (
        SVector(sqrt(a1 * a1 * 3) / 2, a1 / 2),
        SVector(zero(a2), a2),
    )
end

"""
    top(delta)

Return the shortest axial delta under hexagonal periodic boundary conditions.
"""
function (lt::HexagonalTopology)(d::D) where {D<:DeltaCoordinate{2}}
    ps = whichperiodic(lt)
    best = d
    best_dist2 = delta_distance2(lt, d)

    # Skew primitive vectors mean a coupled image search is needed; per-axis
    # wrapping alone can miss the nearest periodic image.
    for j_shift in (ps[2] ? (-1:1) : (0:0))
        for i_shift in (ps[1] ? (-1:1) : (0:0))
            shifted = DeltaCoordinate(
                d[1] + i_shift * size(lt, 1),
                d[2] + j_shift * size(lt, 2),
            )
            shifted_dist2 = delta_distance2(lt, shifted)
            if shifted_dist2 < best_dist2
                best = shifted
                best_dist2 = shifted_dist2
            end
        end
    end
    return best
end

"""
    in(coord, top)

Return whether `coord` lies inside a non-periodic hexagonal topology.
"""
function Base.in(coord, lt::HexagonalTopology{NonPeriodic})
    return all(1 .<= coord .<= size(lt))
end

"""
    in(coord, top)

Return true for a fully periodic hexagonal topology because all coordinates wrap.
"""
@inline Base.in(coord, lt::T) where {T<:HexagonalTopology{Periodic}} = true

"""
    in(coord, top)

Return whether `coord` lies inside all non-periodic axes of a partly periodic
hexagonal topology.
"""
function Base.in(coord, lt::T) where {P<:PartPeriodic,T<:HexagonalTopology{P}}
    _isin = true
    for (i, isperiodic) in enumerate(whichperiodic(lt))
        !_isin && break
        isperiodic && continue
        _isin &= (1 <= coord[i] <= size(lt, i))
    end
    return _isin
end
