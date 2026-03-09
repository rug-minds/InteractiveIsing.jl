export Layer, StateSet, LatticeConstants
Layer(args...) = parse_isinglayer(args...)

struct StateSet{T}
    states::T

    function StateSet(states...)
        new{typeof(tuple(states...))}(tuple(states...))
    end
end

struct LatticeConstants{T}
    constants::T
    function LatticeConstants(constants...)
        new{typeof(tuple(constants...))}(tuple(constants...))
    end
end

"""
Expect in order

args:
    - dims... integers

    - (OPTIONAL) stateset::Tuple
        -> DEFAULT: (-1, 1)
    - (OPTIONAL) Weightgenerator    
        -> DEFAULT: nothing
    - (OPTIONAL) Topology           
        -> DEFAULT: SquareLattice
    - (OPTIONAL) Coords(c1,c2...) typed
        -> DEFAULT: nothing
    - (OPTIONAL) statetype: Discrete() or Continuous()
        -> DEFAULT: Discrete()
    - (OPTIONAL) LatticeConstants(d1, d2, ...) for distance calculations in weightgenerators
        -> DEFAULT: 1.0 for each dimension
    
"""
function parse_isinglayer(args...; periodic = true)
    size_idxs = 1:findlast(x -> x isa Integer, args)
    size = tuple(args[size_idxs]...)
    args = remove_parsed_args(args, size_idxs[end])

    # Parse optional stateset
    stateset_idx = findfirst(x -> x isa StateSet, args)
    stateset = stateset_idx |> isnothing ? (-1, 1) : args[stateset_idx].states
    args = remove_optional_parsed_arg(args, stateset_idx)

    # Parse optional WeightGenerator
    weightgen_idx = findfirst(x -> x isa WeightGenerator, args)
    weightgen = weightgen_idx |> isnothing ? nothing : args[weightgen_idx]
    args = remove_optional_parsed_arg(args, weightgen_idx)

    # Parse optional Lattice Constants
    latconst_idx = findfirst(x -> x isa LatticeConstants, args)
    latconst = latconst_idx |> isnothing ? nothing : args[latconst_idx].constants
    args = remove_optional_parsed_arg(args, latconst_idx)
    if isnothing(latconst)
        ds = tuple(fill(1.0, length(size))...)
    else
        ds = latconst
    end

    # Parse optional Topology
    topology_idx = findfirst(x -> x isa AbstractLayerTopology, args)
    topology = topology_idx |> isnothing ? SquareTopology(size; periodic) : args[topology_idx]
    setdist!(topology, ds)
    args = remove_optional_parsed_arg(args, topology_idx)

    # Parse optional Coords
    coords_idx = findfirst(x -> x isa Coords, args)
    coords = coords_idx |> isnothing ? Coords(nothing) : args[coords_idx]
    args = remove_optional_parsed_arg(args, coords_idx)

    #Parse statetype:
    stype_idx = findfirst(x -> x isa StateType, args)
    stype = stype_idx |> isnothing ? Discrete() : args[stype_idx]
    args = remove_optional_parsed_arg(args, stype_idx)


    # Check for unrecognized arguments
    if !isempty(args)
        throw(ArgumentError("Unrecognized arguments: $(args)"))
    end

    return IsingLayerData(size, stype, stateset, weightgen, topology, coords)
end



