using LoopVectorization, BenchmarkTools

struct Wrap{T} <: DenseArray{T, 1}
    vec::Vector{T}
end

Base.size(w::Wrap) = size(w.vec)
@inline Base.getindex(w::Wrap{T}, i) where T = w.vec[i]
Base.setindex!(w::Wrap, v, i) = (w.vec[i] = v)
Base.length(w::Wrap) = length(w.vec)
Base.eachindex(w::Wrap) = Base.OneTo(length(w.vec))
Base.push!(w::Wrap, v) = push!(w.vec, v)
Base.eltype(w::Wrap) = eltype(w.vec)
Base.firstindex(w::Wrap) = firstindex(w.vec)
Base.lastindex(w::Wrap) = lastindex(w.vec)
Base.iterate(w::Wrap, i) = iterate(w.vec, i)
# Define strides for Wrap to support strided memory access
# Base.pointer(w::Wrap{T}) where T = pointer(w.vec)
# Base.strides(w::Wrap) = strides(w.vec)

# Overload check_args to return true for Wrap
LoopVectorization.check_args(w::Wrap) = true

function Wrap(vec::Vector{T}) where T
    Wrap{T}(vec)
end

const v1 = rand(10^6)
const v2 = rand(10^6)
const w1 = Wrap(v1)
const w2 = Wrap(v2)

function testturbo(w1::Wrap{T}, w2::Wrap{T}) where T
    cum = 0.0
    @turbo for i in eachindex(w1)
        cum += w1.vec[i]
    end
    return cum
end
testturbo(w1, w2)
@benchmark testturbo(w1, w2)