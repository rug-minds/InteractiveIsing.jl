export LatticeType, Square, Rectangular, Oblique, Hexagonal, Rhombic, AnyLattice, AbstractLayerTopology
export AbstractLatticeLayout, DirectLatticeLayout, ZigZagRows, ZigZagColumns
export LatticeTopology, primitive_vectors, covectors, latticetype

struct Square <: LatticeType end
struct Rectangular <: LatticeType end
struct Oblique <: LatticeType end
struct Hexagonal <: LatticeType end
struct Rhombic <: LatticeType end
struct AnyLattice <: LatticeType end

abstract type AbstractLatticeLayout end

"""
Direct coordinate layout for `LatticeTopology`.

Coordinate deltas map linearly onto the primitive vectors, so `[N, 0]` means
`N` steps along the first primitive vector.
"""
struct DirectLatticeLayout <: AbstractLatticeLayout end

"""
Row-staggered coordinate layout for `LatticeTopology`.

Rows advance along the first primitive vector. Every other row is shifted by
half of the second primitive vector, producing the usual zigzag row layout.
For three-dimensional topologies the third coordinate is left unstaggered.
"""
struct ZigZagRows <: AbstractLatticeLayout end

"""
Column-staggered coordinate layout for `LatticeTopology`.

Columns advance along the second primitive vector. Every other column is shifted
by half of the first primitive vector. For three-dimensional topologies the
third coordinate is left unstaggered.
"""
struct ZigZagColumns <: AbstractLatticeLayout end

"""
    LatticeTopology([size], vecs...; layout = DirectLatticeLayout(), periodic = true, origin = nothing, precision = nothing, lattice_type = AnyLattice)

General two- or three-dimensional lattice topology backed by primitive vectors
and reciprocal covectors. The `Layout` type parameter controls whether layer
coordinates map directly to primitive-vector multiples or use a staggered
zigzag embedding. Omitting `size` creates a topology template that is sized by
`sizeto` when it is inserted into a layer.
"""
mutable struct LatticeTopology{
    Kind<:LatticeType,
    Layout<:AbstractLatticeLayout,
    U,
    Dim,
    P<:AbstractFloat,
    V,
    C,
    O,
} <: AbstractLayerTopology{U,Dim}
    pvecs::V
    covecs::C
    size::NTuple{Dim,Int}
    origin::O
end

"""
    LatticeTopology([size], vecs...; kwargs...)

Construct a general lattice topology from `Dim` primitive vectors. If `size` is
omitted, the topology is created as an unsized template. `layout` may be a layout
object, a layout type, or one of `:direct`, `:zigzag_rows`, or `:zigzag_columns`.
"""
function LatticeTopology(
    size::NTuple{D,<:Integer},
    vecs...;
    primitive_vectors = nothing,
    layout = DirectLatticeLayout(),
    periodic::Union{Bool,<:Tuple,Nothing} = true,
    origin = nothing,
    precision = nothing,
    lattice_type::Type{<:LatticeType} = AnyLattice,
    physical_scales = nothing,
) where {D}
    if _lattice_unsized_vector_call(size, vecs, primitive_vectors)
        return LatticeTopology(
            (size, vecs...);
            primitive_vectors,
            layout,
            periodic,
            origin,
            precision,
            lattice_type,
            physical_scales = physical_scales,
        )
    end
    raw_vecs = _lattice_vector_args(Val(D), vecs, primitive_vectors)
    return _lattice_topology(
        ntuple(i -> Int(size[i]), Val(D));
        raw_vecs,
        layout,
        periodic,
        origin,
        precision,
        lattice_type,
        physical_scales = physical_scales,
    )
end

function LatticeTopology(
    vecs...;
    primitive_vectors = nothing,
    layout = DirectLatticeLayout(),
    periodic::Union{Bool,<:Tuple,Nothing} = true,
    origin = nothing,
    precision = nothing,
    lattice_type::Type{<:LatticeType} = AnyLattice,
    physical_scales = nothing,
)
    raw_vecs = _lattice_vector_args_without_size(vecs, primitive_vectors)
    D = _lattice_dimension(raw_vecs, origin)
    raw_vecs = isnothing(raw_vecs) ? nothing : _lattice_vector_args(Val(D), raw_vecs, nothing)
    return _lattice_topology(
        ntuple(_ -> 0, Val(D));
        raw_vecs,
        layout,
        periodic,
        origin,
        precision,
        lattice_type,
        physical_scales = physical_scales,
    )
end

"""
    _lattice_unsized_vector_call(first_arg, rest_args, primitive_vectors)

Disambiguate unsized calls like `LatticeTopology((1, 0), (0, 1))` from sized
calls. If the number of tuple arguments equals the tuple length, interpret them
as primitive vectors.
"""
function _lattice_unsized_vector_call(first_arg::Tuple, rest_args::Tuple, primitive_vectors)
    isnothing(primitive_vectors) || return false
    isempty(rest_args) && return false
    length(rest_args) == length(first_arg) - 1 || return false
    all(arg -> arg isa Tuple && length(arg) == length(first_arg), rest_args) || return false
    return true
end

"""
    _lattice_topology(size, raw_vecs; kwargs...)

Build a `LatticeTopology` after size and vector inputs have been normalized.
"""
function _lattice_topology(
    size::NTuple{D,Int};
    raw_vecs = nothing,
    layout = DirectLatticeLayout(),
    periodic::Union{Bool,<:Tuple,Nothing} = true,
    origin = nothing,
    precision = nothing,
    lattice_type::Type{<:LatticeType} = AnyLattice,
    physical_scales = nothing,
) where {D}
    @assert D in (2, 3) "LatticeTopology currently supports only two- and three-dimensional layers"
    length_scale = _length_reference_scale(physical_scales, raw_vecs, origin)
    raw_vecs = _internal_length_container(raw_vecs, length_scale)
    origin = _internal_length_container(origin, length_scale)
    layout_instance = _lattice_layout(layout)
    P = _lattice_precision(raw_vecs, origin, precision)
    pvecs = _lattice_primitive_vectors(Val(D), P, raw_vecs)
    covecs = _lattice_covectors(pvecs)
    origin_vec = _lattice_origin(Val(D), P, origin)
    U = _lattice_periodicity_type(periodic)
    Layout = typeof(layout_instance)

    return LatticeTopology{
        lattice_type,
        Layout,
        U,
        D,
        P,
        typeof(pvecs),
        typeof(covecs),
        typeof(origin_vec),
    }(pvecs, covecs, size, origin_vec)
end

"""
    _lattice_layout(layout)

Normalize public layout arguments into singleton layout values.
"""
function _lattice_layout(layout::L) where {L<:AbstractLatticeLayout}
    return layout
end

"""
    _lattice_layout(layout_type)

Instantiate a lattice layout passed as a type.
"""
function _lattice_layout(::Type{L}) where {L<:AbstractLatticeLayout}
    return L()
end

"""
    _lattice_layout(layout_symbol)

Map compact symbol names to lattice layout values.
"""
function _lattice_layout(layout::Symbol)
    layout === :direct && return DirectLatticeLayout()
    layout === :zigzag_rows && return ZigZagRows()
    layout === :zigzag_columns && return ZigZagColumns()
    throw(ArgumentError("Unknown lattice layout $layout. Use :direct, :zigzag_rows, or :zigzag_columns."))
end

"""
    _lattice_vector_args(dim, positional_vecs, keyword_vecs)

Resolve primitive vectors from positional or keyword constructor arguments.
"""
function _lattice_vector_args(::Val{D}, vecs::Tuple, primitive_vectors) where {D}
    if !isnothing(primitive_vectors)
        isempty(vecs) || throw(ArgumentError("Pass primitive vectors either positionally or by keyword, not both."))
        @assert length(primitive_vectors) == D "primitive_vectors must contain one vector per topology dimension"
        return tuple(primitive_vectors...)
    elseif isempty(vecs)
        return nothing
    elseif length(vecs) == 1 && vecs[1] isa Tuple && length(vecs[1]) == D
        return vecs[1]
    else
        @assert length(vecs) == D "LatticeTopology needs either zero vectors or one vector per topology dimension"
        return vecs
    end
end

"""
    _lattice_vector_args_without_size(positional_vecs, keyword_vecs)

Resolve primitive vectors for an unsized lattice topology template.
"""
function _lattice_vector_args_without_size(vecs::Tuple, primitive_vectors)
    if !isnothing(primitive_vectors)
        isempty(vecs) || throw(ArgumentError("Pass primitive vectors either positionally or by keyword, not both."))
        return tuple(primitive_vectors...)
    elseif isempty(vecs)
        return nothing
    elseif length(vecs) == 1 && vecs[1] isa Tuple && all(v -> v isa Tuple, vecs[1])
        return vecs[1]
    else
        return vecs
    end
end

"""
    _lattice_dimension(raw_vecs, origin)

Infer the dimension of an unsized lattice topology template.
"""
function _lattice_dimension(raw_vecs, origin)
    if !isnothing(raw_vecs)
        D = length(raw_vecs)
        @assert D in (2, 3) "LatticeTopology currently supports only two- and three-dimensional layers"
        all(v -> length(v) == D, raw_vecs) ||
            throw(ArgumentError("Each primitive vector must have length $D for a $D-dimensional LatticeTopology."))
        return D
    elseif !isnothing(origin)
        D = length(origin)
        @assert D in (2, 3) "LatticeTopology currently supports only two- and three-dimensional layers"
        return D
    else
        throw(ArgumentError("Unsized LatticeTopology needs primitive vectors or an origin to infer dimension."))
    end
end

"""
    _lattice_precision(raw_vecs, origin, precision)

Choose the floating-point precision used by lattice vectors and coordinates.
"""
function _lattice_precision(raw_vecs, origin, precision)
    !isnothing(precision) && return precision

    P = Float32
    if !isnothing(raw_vecs)
        for vec in raw_vecs
            P = promote_type(P, typeof(float(first(vec))))
        end
    end
    if !isnothing(origin)
        P = promote_type(P, typeof(float(first(origin))))
    end
    return P
end

"""
    _lattice_primitive_vectors(dim, precision, raw_vecs)

Convert primitive-vector inputs into a statically sized tuple of `SVector`s.
"""
function _lattice_primitive_vectors(::Val{D}, ::Type{P}, raw_vecs) where {D,P}
    if isnothing(raw_vecs)
        return ntuple(Val(D)) do i
            SVector{D,P}(ntuple(j -> i == j ? one(P) : zero(P), Val(D)))
        end
    end

    return ntuple(Val(D)) do i
        @assert length(raw_vecs[i]) == D "Primitive vector $i has length $(length(raw_vecs[i])); expected $D"
        SVector{D,P}(ntuple(j -> P(raw_vecs[i][j]), Val(D)))
    end
end

"""
    _lattice_covectors(pvecs)

Return covectors dual to the primitive-vector basis.
"""
function _lattice_covectors(pvecs::NTuple{D,SVector{D,P}}) where {D,P}
    basis = reduce(hcat, pvecs)
    invbasis = inv(basis)
    return ntuple(Val(D)) do i
        SVector{D,P}(ntuple(j -> P(invbasis[i, j]), Val(D)))
    end
end

"""
    _lattice_origin(dim, precision, origin)

Convert an optional origin into a statically sized vector.
"""
function _lattice_origin(::Val{D}, ::Type{P}, origin) where {D,P}
    if isnothing(origin)
        return SVector{D,P}(ntuple(_ -> zero(P), Val(D)))
    end
    @assert length(origin) == D "origin must have one entry per topology dimension"
    return SVector{D,P}(ntuple(i -> P(origin[i]), Val(D)))
end

"""
    _lattice_periodicity_type(periodic)

Encode public periodicity arguments as a concrete periodicity type.
"""
function _lattice_periodicity_type(periodic::Bool)
    return periodic ? Periodic : NonPeriodic
end

"""
    _lattice_periodicity_type(periodic)

Treat `nothing` as fully periodic for consistency with existing topology constructors.
"""
function _lattice_periodicity_type(::Nothing)
    return Periodic
end

"""
    _lattice_periodicity_type(periodic_axes)

Encode a tuple of periodic axes as `PartPeriodic`.
"""
function _lattice_periodicity_type(periodic_axes::Tuple)
    return PartPeriodic(periodic_axes...)
end

"""
    primitive_vectors(top)

Return the primitive vectors stored by a general lattice topology.
"""
@inline function primitive_vectors(top::T) where {T<:LatticeTopology}
    return top.pvecs
end

"""
    covectors(top)

Return the reciprocal covectors stored by a general lattice topology.
"""
@inline function covectors(top::T) where {T<:LatticeTopology}
    return top.covecs
end

"""
    origin(top)

Return the world-space origin of a general lattice topology.
"""
@inline function origin(top::T) where {T<:LatticeTopology}
    return top.origin
end

"""
    lattice_constants(top)

Return the current primitive-vector lengths for a general lattice topology.
"""
function lattice_constants(top::T) where {T<:LatticeTopology}
    return map(norm, primitive_vectors(top))
end

"""
    setdist!(top, lattice_constants)

Rescale each primitive vector to the requested length and refresh covectors.
"""
function setdist!(top::T, lattice_constants::NTuple{D}) where {Kind,Layout,U,D,P,T<:LatticeTopology{Kind,Layout,U,D,P}}
    length_scale = _length_reference_scale(nothing, lattice_constants)
    lattice_constants = _internal_length_container(lattice_constants, length_scale)
    top.pvecs = ntuple(Val(D)) do i
        old_length = norm(top.pvecs[i])
        old_length == 0 && throw(ArgumentError("Cannot rescale a zero primitive vector."))
        return top.pvecs[i] * (P(lattice_constants[i]) / old_length)
    end
    top.covecs = _lattice_covectors(top.pvecs)
    return top
end

"""
    sizeto(top, size)

Return a `LatticeTopology` with the same geometry and layout sized to a layer.
Unsized templates have zero extents; fully sized topologies must already match.
"""
function sizeto(top::T, layer_size::NTuple{D,<:Integer}) where {Kind,Layout,U,D,P,T<:LatticeTopology{Kind,Layout,U,D,P}}
    requested_size = ntuple(i -> Int(layer_size[i]), Val(D))
    current_size = size(top)
    if current_size == requested_size
        return top
    elseif all(iszero, current_size)
        return LatticeTopology{
            Kind,
            Layout,
            U,
            D,
            P,
            typeof(top.pvecs),
            typeof(top.covecs),
            typeof(top.origin),
        }(top.pvecs, top.covecs, requested_size, top.origin)
    else
        throw(ArgumentError("Explicit topology size $(current_size) does not match layer size $(requested_size)."))
    end
end

"""
    latticetype(top)

Return the lattice-family marker encoded in a `LatticeTopology` type.
"""
@inline function latticetype(::T) where {Kind,Layout,U,D,P,T<:LatticeTopology{Kind,Layout,U,D,P}}
    return Kind
end

"""
    is_translation_invariant(top)

Return whether metric distances depend only on coordinate deltas.
"""
@inline function is_translation_invariant(::T) where {T<:AbstractLayerTopology}
    return true
end

"""
    is_translation_invariant(top)

Row- and column-staggered layouts need the anchor coordinate to compute metric
distances, because the half-step offset alternates with parity.
"""
@inline function is_translation_invariant(::T) where {Kind,U,D,P,T<:LatticeTopology{Kind,ZigZagRows,U,D,P}}
    return false
end

@inline function is_translation_invariant(::T) where {Kind,U,D,P,T<:LatticeTopology{Kind,ZigZagColumns,U,D,P}}
    return false
end

"""
    _layout_coefficients(layout, coord)

Return coordinate coefficients before multiplication by primitive vectors.
"""
function _layout_coefficients(::L, coord::C) where {D,L<:DirectLatticeLayout,C<:Coordinate{D}}
    return ntuple(i -> coord[i], Val(D))
end

"""
    _layout_coefficients(layout, coord)

Return row-staggered coefficients for two- and three-dimensional coordinates.
"""
function _layout_coefficients(::L, coord::C) where {D,L<:ZigZagRows,C<:Coordinate{D}}
    @assert D in (2, 3) "ZigZagRows supports only two- and three-dimensional coordinates"
    shift = isodd(coord[1] - 1) ? 0.5 : 0.0
    return ntuple(i -> i == 2 ? coord[i] + shift : coord[i], Val(D))
end

"""
    _layout_coefficients(layout, coord)

Return column-staggered coefficients for two- and three-dimensional coordinates.
"""
function _layout_coefficients(::L, coord::C) where {D,L<:ZigZagColumns,C<:Coordinate{D}}
    @assert D in (2, 3) "ZigZagColumns supports only two- and three-dimensional coordinates"
    shift = isodd(coord[2] - 1) ? 0.5 : 0.0
    return ntuple(i -> i == 1 ? coord[i] + shift : coord[i], Val(D))
end

"""
    _lattice_world_position(top, coord; wrap_coordinate = true)

Map a topology coordinate into world coordinates without allocating dynamic
arrays. `wrap_coordinate = false` is used for periodic image searches.
"""
function _lattice_world_position(
    top::T,
    coord::C;
    wrap_coordinate::Bool = true,
) where {Kind,Layout,U,D,P,T<:LatticeTopology{Kind,Layout,U,D,P},C<:Coordinate{D}}
    c = wrap_coordinate ? wrap(top, coord) : coord
    coefficients = _layout_coefficients(Layout(), c)
    position = origin(top)

    # Accumulate in world space so all layouts share the same vector backend.
    for i in 1:D
        position += P(coefficients[i]) * top.pvecs[i]
    end
    return position
end

"""
    in(coord, top)

Return whether `coord` lies inside a non-periodic general lattice topology.
"""
function Base.in(coord, top::LatticeTopology{Kind,Layout,NonPeriodic}) where {Kind,Layout}
    return all(1 .<= coord .<= size(top))
end

"""
    in(coord, top)

Return true for a fully periodic general lattice topology because all
coordinates wrap.
"""
@inline function Base.in(coord, top::T) where {Kind,Layout,T<:LatticeTopology{Kind,Layout,Periodic}}
    return true
end

"""
    in(coord, top)

Return whether `coord` lies inside all non-periodic axes of a partly periodic
general lattice topology.
"""
function Base.in(coord, top::T) where {Kind,Layout,P<:PartPeriodic,T<:LatticeTopology{Kind,Layout,P}}
    _isin = true
    for (i, isperiodic) in enumerate(whichperiodic(top))
        !_isin && break
        isperiodic && continue
        _isin &= (1 <= coord[i] <= size(top, i))
    end
    return _isin
end

# Return the shortest coordinate delta for translation-invariant lattice layouts.
# Staggered layouts need the anchor coordinate, so the raw delta is unchanged.
function (top::LatticeTopology{Kind,Layout,U,D,P})(delta::DC) where {Kind,Layout,U,D,P,DC<:DeltaCoordinate{D}}
    is_translation_invariant(top) || return delta

    best = delta
    best_dist2 = delta_distance2(top, delta)
    ranges = ntuple(i -> periodic(U(), Val(i)) ? (-1:1) : (0:0), Val(D))

    for shift_ci in CartesianIndices(ranges)
        shifted = DeltaCoordinate(ntuple(i -> delta[i] + shift_ci[i] * size(top, i), Val(D)))
        shifted_dist2 = delta_distance2(top, shifted)
        if shifted_dist2 < best_dist2
            best = shifted
            best_dist2 = shifted_dist2
        end
    end
    return best
end
