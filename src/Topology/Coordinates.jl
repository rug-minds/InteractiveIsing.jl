
export Coordinate, DeltaCoordinate, coord, norm2, delta

struct Coordinate{N,T} <: Base.AbstractCartesianIndex{N} # N dimensional coordinate
    top::T
    coords::CartesianIndex{N}

    function Coordinate(top::LayerTopology, n::Integer...; check = true)
        dims = length(size(top))
        @assert length(n) == dims "Number of coordinates $(length(n)) must match topology dimensions $dims"

        periodic = whichperiodic(top)
        new_coords = ntuple(i -> periodic[i] ? mod1(n[i], size(top)[i]) : n[i], length(n))
        
        # @assert all(x -> 0 < x[1] <= size(top)[x[2]], zip(new_coords, 1:length(n))) "Coordinate $n is out of bounds for topology of size $(size(top))"
        if check
            assert_gen = [0 < new_coords[i] <= size(top)[i] for i in 1:length(n)]::Vector{Bool}
            @assert all(assert_gen) "Coordinate $n is out of bounds for topology of size $(size(top))"
        end

        new{dims, typeof(top)}(top, CartesianIndex(new_coords))
    end
end

Coordinate(top, t::Tuple) = Coordinate(top, t...)
Coordinate(top, ci::CartesianIndex) = Coordinate(top, ci.I...)
Coordinate(top::LayerTopology, i::Int) = Coordinate(top, CartesianIndices(size(top))[i])
convert(::Type{<:CartesianIndex}, c::Coordinate) = c.coords

offset(c::Coordinate, deltas...; check = false) = Coordinate(c.top, ntuple(i->c.coords[i]+deltas[i], length(c))...; check = false)

function Base.:(-)(c1::Coordinate, c2::Coordinate)
    @assert c1.top == c2.top "Coordinates must belong to the same topology"

    # Coordinate(c1.top, ntuple(i->c2.coords[i]-c1.coords[i], length(c1.coords))...)
    delta(c1, c2)
end
top(c::Coordinate) = c.top
Base.abs(c::Coordinate) = Coordinate(c.top, abs.(c.coords)...)

Base.getindex(c::Coordinate, i::Int) = c.coords[i]
Base.iterate(c::Coordinate, state = 1) = iterate(c.coords, state)
Base.length(c::Coordinate{N}) where N = N
Base.size(c::Coordinate{N}) where N = (length(c),)

function delta(c1::Coordinate, c2::Coordinate)
    @assert c1.top == c2.top "Coordinates must belong to the same topology"
    DeltaCoordinate(c1, ntuple(i->c2.coords[i]-c1.coords[i], length(c1.coords)), top = c1.top)
end

function delta(top::LayerTopology, ci1::CartesianIndex, ci2::CartesianIndex)
    DeltaCoordinate(Coordinate(top, ci1), ntuple(i->ci2[i]-ci1[i], length(ci1)), top = top)
end

coord(x...) = Coordinate(x...)

struct DeltaCoordinate{N,T} <: AbstractVector{Int} # N dimensional coordinate difference
    start::Coordinate{N,T}
    deltas::NTuple{N, Int}
    top::T
end

Base.getindex(dc::DeltaCoordinate, i::Int) = dc.deltas[i]
Base.iterate(dc::DeltaCoordinate, state = 1) = iterate(dc.deltas, state)
Base.length(dc::DeltaCoordinate{N}) where N = N
Base.size(dc::DeltaCoordinate{N}) where N = (N,)
Base.size(dc::DeltaCoordinate, i::Int) = size(dc)[i]

function DeltaCoordinate(c1::Coordinate, ds; top = SquareTopology(periodic=false))
    axisisperiodic = whichperiodic(c1.top)
    # Get the shortest ds
    ds = ntuple(i -> axisisperiodic[i] ? begin
            d = ds[i]
            halfsize = div(size(c1.top)[i], 2)
            if abs(d) > halfsize
                d - sign(d) * size(c1.top)[i]
            else
                d
            end
        end : ds[i], length(ds))
    DeltaCoordinate{length(ds), typeof(top)}(c1, ds, top)
end

LinearAlgebra.norm(dc::DeltaCoordinate) = sqrt(norm2(dc))
norm2(dc::DeltaCoordinate) = sum(x -> x^2, dc.deltas)
function Base.:^(dc::DeltaCoordinate, p::Integer)
    return sum(x -> x^p, dc.deltas)
end

