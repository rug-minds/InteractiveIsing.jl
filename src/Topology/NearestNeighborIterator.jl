export NearestNeighborCandidate, nearest_neighbor_iterator

"""
    NearestNeighborCandidate(offset, dc, dr, has_dr)

Represent one topology-provided candidate edge offset for adjacency
construction. `offset` is the coordinate-space displacement used to find the
candidate endpoint, `dc` is the topology-adjusted displacement passed to weight
generators, and `dr` is a precomputed metric distance when `has_dr` is true.
"""
struct NearestNeighborCandidate{D,R<:Real}
    offset::NTuple{D,Int}
    dc::DeltaCoordinate{D}
    dr::R
    has_dr::Bool
end

"""
    nearest_neighbor_iterator(topology, NN, layer_size = size(topology))

Return a reusable lazy iterator of topology-specific candidate neighbor offsets.
The default implementation preserves the legacy Cartesian stencil
`[-NN, NN]^D \\ {0}`, then lets the weight generator filter candidates by the
topology distance `dr`.

Topology packages should extend this method next to the topology type they add.
For example, a Chimera topology can provide hardware-native coupler candidates
without changing adjacency construction code.
"""
function nearest_neighbor_iterator(
    topology::T,
    NN::NTuple{D,<:Integer},
    layer_size::NTuple{D,<:Integer} = size(topology),
) where {U,D,T<:AbstractLayerTopology{U,D}}
    ranges = ntuple(i -> (-Int(NN[i])):Int(NN[i]), Val(D))
    periodic_axes = whichperiodic(topology)
    sized = ntuple(i -> Int(layer_size[i]), Val(D))
    translation_invariant = is_translation_invariant(topology)

    # This generator is intentionally lazy: topology-specific methods can
    # replace the candidate source without changing sparse adjacency filling.
    return (
        _nearest_neighbor_candidate(topology, sized, periodic_axes, translation_invariant, offset_ci.I)
        for offset_ci in CartesianIndices(ranges)
        if !all(iszero, offset_ci.I)
    )
end

"""
    _nearest_neighbor_candidate(topology, layer_size, periodic_axes, translation_invariant, offset)

Build one default Cartesian-stencil neighbor candidate.
"""
function _nearest_neighbor_candidate(
    topology::T,
    layer_size::NTuple{D,<:Integer},
    periodic_axes::NTuple{D,Bool},
    translation_invariant::Bool,
    delta_offset::NTuple{D,<:Integer},
) where {U,D,T<:AbstractLayerTopology{U,D}}
    offset = ntuple(i -> Int(delta_offset[i]), Val(D))
    wrapped_offset = ntuple(Val(D)) do i
        di = offset[i]
        if periodic_axes[i]
            halfsize = layer_size[i] >>> 1
            abs(di) > halfsize && (di -= sign(di) * layer_size[i])
        end
        di
    end
    dc = DeltaCoordinate(wrapped_offset)
    dr = translation_invariant ? delta_distance(topology, dc) : NaN
    return NearestNeighborCandidate(offset, dc, dr, translation_invariant)
end

"""
    _candidate_distance(candidate, topology, c1, c2)

Return the candidate's precomputed distance, or compute the anchored topology
distance when the candidate depends on the source coordinate.
"""
@inline function _candidate_distance(
    candidate::NearestNeighborCandidate,
    topology::T,
    c1::C1,
    c2::C2,
) where {T<:AbstractLayerTopology,C1<:Coordinate,C2<:Coordinate}
    return candidate.has_dr ? candidate.dr : dist(topology, c1, c2)
end
