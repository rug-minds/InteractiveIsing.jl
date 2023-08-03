using BenchmarkTools
using LinearAlgebra
abstract type FunctionType end

struct F1 <: FunctionType end
struct F2 <: FunctionType end

try 
    mutable struct Outer
        num::Float64
        funcType::Type{T} where T <: FunctionType
        i::Inner
    
        function Outer(num, ft)
            o = new(num,ft)
            o.i = Inner(o, ([1,2],[2,3]))
            return o
        end
    end
catch
end

mutable struct Inner
    outer::Outer
    vecs::Tuple{Vector{Float64}, Vector{Float64}}
end

x(i::Inner) = i.vecs[1]
y(i::Inner) = i.vecs[2]

mutable struct Outer
    num::Float64
    funcType::Type{T} where T <: FunctionType
    i::Inner

    function Outer(num, ft)
        o = new(num,ft)
        o.i = Inner(o, ([1,2],[2,3]))
        return o
    end
end

function normInner(x, y, num, ::Type{F1})
    return norm(x) + norm(y) + num^2
end

function normInner(x, y, num, ::Type{F2})
    return norm(x)^2 + norm(y)^2 + num
end

addInner(i::Inner, num, ft::Type{FT}) where FT <: FunctionType = normInner(x(i), y(i), num, ft) 

addOuter(outer::Outer) = addInner(outer.i, outer.num, outer.funcType)

addInner(i::Inner) = addInner(i, i.outer.num, i.outer.funcType)

const outer1 = Outer(9.99, F1)
const inner1 = outer1.i
const num = outer1.num

@benchmark addInner(inner1, num, F1)
@benchmark addInner(inner1)
@benchmark addInner(outer1.i, outer1.num, outer1.funcType)
@benchmark addOuter(outer1)
# @benchmark outer1.i