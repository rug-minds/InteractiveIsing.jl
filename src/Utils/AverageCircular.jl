mutable struct AverageCircular{T} <: AbstractVector{T}
    data::Vector{T}
    ptr::Int
    sum::T
    len::Int
end

AverageCircular(T::DataType, len::Integer) = AverageCircular{T}(zeros(T,len), 1, zero(T), len)

Base.push!(ac::AverageCircular{T}, val::OT) where {T, OT >: T} = push!(ac, convert(T, val))


function Base.push!(ac::AverageCircular{T}, val::T) where T
    ac.data[ac.ptr] = val
    ac.ptr = mod1(ac.ptr + 1, ac.len)
    currentval = ac.data[ac.ptr]
    ac.sum += val - currentval
    return ac
end

function Base.getindex(ac::AverageCircular{T}, idx::Integer) where T
    return ac.data[mod1(ac.ptr - idx + 1, ac.len)]
end

avg(ac::AverageCircular) = ac.sum/ac.len
Base.length(ac::AverageCircular) = ac.len
Base.size(ac::AverageCircular) = (ac.len,)
function Base.show(io::IO, ac::AverageCircular)
    print(io, "AverageCircular with $(length(ac)) elements")
end


export AverageCircular, avg