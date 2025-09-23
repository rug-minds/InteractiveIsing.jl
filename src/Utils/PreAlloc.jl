abstract type AbstractPreAlloc{T} <: AbstractVector{T} end

mutable struct Prealloc{T} <: AbstractPreAlloc{T}
    const vec::Vector{T} 
    used::Int
    const maxsize::Int
end

function Prealloc(type::Type, N)
        println("Creating Prealloc of type ", type, " with size ", N)
        vec = Vector{type}(undef, N)
        println("Length of vec: ", length(vec))
        maxsize = N
        used = 0
        Prealloc{type}(vec, used, maxsize)
end
struct ThreadedPrealloc{PreallocT}
    vec::Vector{PreallocT}
end
@inline Base.getindex(pre::ThreadedPrealloc, i) = pre.vec[i]

function ThreadedPrealloc(type::Type, N, nthreads)
    PreallocT = typeof(Prealloc(type, N))
    vec = Vector{PreallocT}(undef, nthreads)
    for i in 1:nthreads
        vec[i] = Prealloc(type, N)
    end
    return ThreadedPrealloc(vec)
end

Base.getindex(pre::AbstractPreAlloc, i) = pre.vec[i]
Base.setindex!(pre::AbstractPreAlloc, tup, i) = (pre.vec[i] = tup; pre.used = capacity(pre.used, i))
Base.length(pre::AbstractPreAlloc) = pre.used
Base.push!(pre::AbstractPreAlloc, tup) = (pre.vec[pre.used+1] = tup; pre.used += 1)
reset!(pre::AbstractPreAlloc) = (pre.used = 0; return)
Base.size(pre::AbstractPreAlloc) = tuple(pre.used)
Base.eachindex(pre::AbstractPreAlloc) = Base.OneTo(pre.used)
maxsize(pre::AbstractPreAlloc) = pre.maxsize

alloc_length(pre::AbstractPreAlloc) = length(pre.vec)

export Prealloc

# using StaticArrays

# mutable struct SPrealloc{S,T} <: AbstractPreAlloc{T}
#     const vec::MVector{S,T} 
#     used::Int32
#     const maxsize::Int32

#     function SPrealloc(type::Type, N)
#         vec = MVector{N, type}(Vector{type}(undef, N))
#         maxsize = N
#         used = 0
#         return new{N, type}(vec, used, maxsize)
#     end

# end