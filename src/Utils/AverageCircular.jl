mutable struct AverageCircular{T} <: AbstractVector{T}
    data::Vector{T}
    ptr::Int
    sum::T
    len::Int
end

AverageCircular(T::DataType, len::Integer) = AverageCircular{T}(zeros(T,len), 1, zero(T), len)

function Base.push!(ac::AverageCircular{T}, val::T) where T
    currentval = ac.data[ac.ptr]
    ac.data[ac.ptr] = val
    ac.sum += val - currentval
    ac.ptr = mod1(ac.ptr + 1, ac.len)
    return ac
end

function Base.getindex(ac::AverageCircular{T}, idx::Integer) where T
    return ac.data[mod1(ac.ptr - idx + 1, ac.len)]
end

avg(ac::AverageCircular) = ac.sum/ac.len
length(ac::AverageCircular) = ac.len
size(ac::AverageCircular) = (ac.len,)
function Base.show(io::IO, ac::AverageCircular)
    print(io, "AverageCircular with $(length(ac)) elements")
end


export AverageCircular, avg