using BenchmarkTools
using LinearAlgebra

abstract type FunctionType end

struct F1 <: FunctionType end
struct F2 <: FunctionType end

struct Vecs
    x::Vector{Float64}
    y::Vector{Float64}
end

const vecs = Vecs([1,2],[2,3])


function combine(x, y, ::Type{F1})
    return norm(x) + norm(y)
end

function combine(x, y, ::Type{F2})
    return norm(x)^2 + norm(y)^2
end

# combine(v::Vecs, t::Type{FT}) where FT <: FunctionType = combine(v.x, v.y, t)
# combine(v::Vecs, t) = combine(v.x, v.y, t)
combine(v, t) = combine(v.x, v.y, t)


function test1(v, t) 
    ti = time()
    for _ in 1:10^8
        combine(v, t)
    end
    tf = time()
    return tf-ti
end
# test1(vecs, F1)

function test2(v, t::Type{FT}) where FT <: FunctionType
    ti = time()
    for _ in 1:10^8
        combine(v, t)
    end
    tf = time()
    return tf-ti
end
# test2(vecs, F1)

display(test1(vecs, F1))
display(test2(vecs, F1))
