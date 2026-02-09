using LinearAlgebra

export PeriodicityType, Periodic, NonPeriodic, PartPeriodic
export periodic, periodicaxes
abstract type PeriodicityType end
struct Periodic <: PeriodicityType end
struct PartPeriodic{T} <: PeriodicityType end
parts(P::Type{<:PartPeriodic{Parts}}) where {Parts} = Parts
parts(p::PartPeriodic) = parts(typeof(p))

struct NonPeriodic <: PeriodicityType end

const part_periodic_map = Dict(
    :x => 1,
    :y => 2,
    :z => 3
)

map_pp_f(i::Integer) = i
map_pp_f(symb::Symbol) = part_periodic_map[symb]


function PartPeriodic(args...) 
    # assert only has a combination of x y and z
    @assert all(x -> x in (:x, :y, :z), args) || all(x -> x isa Integer && 1 <= x , args) "PartPeriodic only takes a combination of :x, :y, :z or integers"
    args = tuple(map(map_pp_f, args)...)
    return PartPeriodic{args}
end

periodic(p::PeriodicityType, symb::Symbol) = periodic(p, Val(symb))
periodic(p::PeriodicityType, dim::Int) = periodic(p, Val(dim))

@generated function periodic(P::PartPeriodic{Parts}, ::Val{dim}) where {Parts,dim}
    if dim isa Symbol
        dim = map_pp_f(dim)
    end
    found = findfirst(x -> x == dim, Parts)
    return :($(!isnothing(found)))
end


periodic(P::Periodic, ::Val{symb}) where symb = true
periodic(P::NonPeriodic, ::Val{symb}) where symb = false
periodicaxes(P::PartPeriodic{Parts}, dims) where {Parts} = Parts
periodicaxes(P::Periodic, dims) = ntuple(i -> i, dims)
periodicaxes(P::NonPeriodic, dims) = tuple()



# """
# Why is this here?
# """
# struct GenericTopology{U} <: AbstractLayerTopology{U,0} end

