using BenchmarkTools
using LinearAlgebra
abstract type FunctionType end

struct F1 <: FunctionType end
struct F2 <: FunctionType end

struct Inner
    vecs::Tuple{Vector{Float64}, Vector{Float64}}
end

x(i::Inner) = i.vecs[1]
y(i::Inner) = i.vecs[2]

mutable struct Outer
    i::Inner
    num::Float64
    funcType::Type{T} where T <: FunctionType
end

function normInner(x, y, num, ::Type{F1})
    return norm(x) + norm(y) + num^2
end

function normInner(x, y, num, ::Type{F2})
    return norm(x)^2 + norm(y)^2 + num
end

addInner(i::Inner, num, ft::Type{FT}) where FT <: FunctionType = normInner(x(i), y(i), num, ft) 

addOuter(outer::Outer) = addInner(outer.i, outer.num, outer.funcType)

const outer1 = Outer(Inner(([1,2],[2,3])), 9.99, F1)
const inner1 = outer1.i
const num = outer1.num

@benchmark addInner($inner1, $num, F1)
@benchmark addInner($outer1.i, outer1.num, $outer1.funcType)
@benchmark addOuter(outer1)

function test1(t::Type{FT}) where FT <: FunctionType
    ti = time()
    for i in 1:100000000
        addInner(outer1.i, outer1.num, t)
    end
    tf = time()
    println(tf-ti)
end
test1(outer1.funcType)

function test2()
    ti = time()
    for i in 1:100000000
        addInner(inner1, num, F1)
    end
    tf = time()
    println(tf-ti)
end
test2()