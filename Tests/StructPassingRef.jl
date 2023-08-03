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

struct Outer
    i::Ref{Inner}
    num::Ref{Float64}
    funcType::Ref{Union{Type{F1},Type{F2}}}
end

@inline inn(o::Outer)::Inner = o.i[]
@inline num(o::Outer)::Float64 = o.num[]
@inline ft(o::Outer)::Union{Type{F1},Type{F2}} = o.funcType[]

function normInner(x, y, nmbr, ::Type{F1})
    return norm(x) + norm(y) + nmbr^2
end

function normInner(x, y, nmbr, ::Type{F2})
    return norm(x)^2 + norm(y)^2 + nmbr
end

addInner(i::Inner, nmbr, ft::Type{FT}) where FT <: FunctionType = normInner(x(i), y(i), nmbr, ft) 

addOuter(outer::Outer) = addInner(inn(outer), num(outer), ft(outer))

const outer1 = Outer(
    Ref(Inner(([1,2],[2,3]))), 
    Ref(9.9), 
    Ref{Union{Type{F1},Type{F2}}}(F1))

const inner1 = inn(outer1)
const num1 = num(outer1)

@benchmark addInner($inner1, $num1, F1)
@benchmark addInner($inn(outer1), $num(outer1), $ft(outer1))
@benchmark addOuter(outer1)

function test1()
    ti = time()
    for i in 1:100000000
        addInner(inn(outer1), num(outer1), F!)
    end
    tf = time()
    println(tf-ti)
end
test1()

function test2()
    ti = time()
    for i in 1:100000000
        addInner(inner1, num1, F1)
    end
    tf = time()
    println(tf-ti)
end
test2()