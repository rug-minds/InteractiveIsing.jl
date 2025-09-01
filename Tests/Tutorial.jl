#STRUCT TUTORIAL

mutable struct MyStruct{T} # Struct definition
    x::T
    y::T
end

# Constructor for MyStruct with two parameters
function MyStruct(x::T1, y::T2) where {T1, T2}
    T = promote_type(T1, T2) # Determine the common type
    return MyStruct{T}(T(x), T(y)) # Create an instance of MyStruct with
end

# Constructor for MyStruct with a single type parameter, initializing both fields to zero
function MyStruct(type::DataType) 
    return MyStruct{type}(zero(type), zero(type))
end

getparam(s::MyStruct{T}) where T = T

@inline get_x(s::MyStruct{Float64}) = s.x^2
@inline get_x(s::MyStruct{Int}) = s.x

get_y(s::MyStruct) = s.y



"""
Defines implcitly two functions:
    Compiles the functions inline:
    multvals(s::MyStruct{Float64}) = x^2*y
    multvals(s::MyStruct{Int}) = x*y
"""
function multvals(s::MyStruct) # Generic function to get x
    get_x(s)*get_y(s)
end

struct MyTensor{T, D}
    data::Vector{T}
end


## Function TUTORIAL

"""
Functions are "names"
Methods are the actual implementations of the function

All are define a "method" for the function mysum

Julia tries to find most specific method for the function
"""

# method mysum(a::Float64, b::Float64) for function mysum
function mysum(a::Float64, b::Float64)
    return a + b
end

# method mysum(a::Int, b::Int) for function mysum
function mysum(a::Int, b::Int)
    return a^2 + b^2
end

# defines method mysum(a::Any,b::Any)
function mysum(a,b)
    return nothing
end

"""
Next two function create problem with ambiguity
"""

function mysum(a::Int, b)
    #somecode
end

function mysum(a, b::Int)
    #somecode
end

#### TUPLE TYPES TUTORIAL

# normal tuple
t1 = (1, 2.0, "three", [1,2,3], (4, 5))

# NamedTuple
t2 = (a = 1, b = 2.0, c = "three", d = [1,2,3], e = (4, 5))

typeof(t1) # Returns Tuple{Int, Float64, String, Vector{Int}, Tuple{Int, Int}}
typeof(t2) # Returns NamedTuple{(:a, :b, :c, :d, :e), Tuple{Int, Float64, String, Vector{Int}, Tuple{Int, Int}}}

"""
Next two function basically equivalent if a = t[1], b = t[2], c = t[3], d = t[4], e = t[5]
    Because tuple also has a type where the parameters are defined
    The set of types of parameters is equivalent to the signature of the function "nontuplefunc"
"""
function nontuplefunc(a,b,c,d,e)
    #somcode
    return a
end

function tuplefunc(t::Tuple)
    a = t[1]
    b = t[2]
    c = t[3]
    d = t[4]
    e = t[5]
    return a
end

"""
Again equivalent
"""
function namedtuplefunc(t::NamedTuple)
    (;a, b, c, d, e) = t
    #samecode
    return a
end

# Now with functions in body

function practicalcode1(a,c,e)
    #somecode
    return a
end
function practicalcode2(a,b, c,e)
    #somecode
    return b
end

# Need to pass every name every time
# This is not flexible, but works if you know exactly which functions you want to call
function nontuplefunc(a,b,c,d,e)
    practicalcode1(a, c, e)
    practicalcode2(a, b, c, e)
    return a
end

## FOR TUPLES
# Ordering of tuple parameters is important
function practicalcodetuple1(t::Tuple)
    a = t[1]
    c = t[3]
    e = t[5]
    return a
end

function practicalcodetuple2(t::Tuple)
    b = t[2]
    d = t[4]
    return b
end

function tuplefunc(t::Tuple)
    practicalcodetuple1(t)
    practicalcodetuple2(t)
    return a
end

## FOR NAMEDTUPLES

function practicalcodenamedtuple1(t::NamedTuple)
    (;a, c, e) = t
    return a
end

function practicalcodenamedtuple2(t::NamedTuple)
    (;b, d) = t
    return b
end

function namedtuplefunc(t::NamedTuple)
    practicalcodenamedtuple1(t)
    practicalcodenamedtuple2(t)
    return a
end


# New example with higher order functions

# Non-tuple
normalalgorithm(a, b, c) = a + c
newalgo(a,c) = a * c # WONT WORK as general_algorithm(a,b,c, newalgo) because b is not used

function general_algorithm(a,b,c, algorithm::Function)
    # some code
    result = algorithm(a, b, c)
    # some more code
    return result
end

#Tuple
# All types are known so still typestable since t has full signature
tuplealgorithm(t::Tuple) = t[1] + t[3]
newtuplealgo(t::Tuple) = t[1] * t[3] # Will work, but ordering needs to be consistent

function general_algorithm_tuple(t::Tuple, algorithm::Function)
    # some code
    result = algorithm(t)
    # some more code
    return result
end

# NamedTuple
# All types are known so still typestable since t has full signature
namedtuplealgorithm(t::NamedTuple) = t.a + t.c
newnamedtuplealgo(t::NamedTuple) = t.a * t.c # Will work, but names need to be consistent

function general_algorithm_namedtuple(t::NamedTuple, algorithm::Function)
    # some code
    result = algorithm(t)
    # some more code
    return result
end

getargs()

# STRUCT TUTORIAL 2: Setters and getters

"""
First definition
struct SomeStruct{T}
    x::T
    y::T
end

getx(s::SomeStruct) = s.x
gety(s::SomeStruct) = s.y

Comes with a function
function somefunction(s::SomeStruct)
    return s.x + s.y
end

function otherplusfunction(s::SomeStruct)
    return getx(s) + gety(s)
end

function otherprodfunction(s::SomeStruct)
    return getx(s) * gety(s)
end

"""

# NEW definition
# BREAKS SOMEFUNCTION
struct SomeStruct{N, T}
    data::NTuple{N, T}
end

function SomeStruct(data::T...) where T
    SomeStruct{length(data), T}(data)
end

"""
I could make 
SomeStruct with x::T and y::T a special case with SomeStruct{2, T} and data::NTuple{2, T}

I still have support for the old functions
"""

s1 = SomeStruct(1,2,3,5)

# Redefine getters
# otherplusfunction AND otherprodfunction still work
# Only need to redefine the getters
getx(s::SomeStruct) = s.data[1]
gety(s::SomeStruct) = s.data[2]
