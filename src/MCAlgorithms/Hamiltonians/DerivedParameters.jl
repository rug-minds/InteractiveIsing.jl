export StateLike

abstract type DerivedParameter end

struct StateLike{T,F} <: DerivedParameter
    default_el::F
end

StateLike(T, default_el = 0) = StateLike{T, typeof(default_el)}(default_el)

# Vector(val, size...) = fill(val, size...)

function (ss::StateLike{T})(g::AbstractIsingGraph) where T
    s = state(g)
    return filltype(T, ss.default_el, size(s)...)
end

struct FromGraph{F} <: DerivedParameter 
    f::F
end

function (fg::FromGraph{F})(g::AbstractIsingGraph) where F
    return fg.f(g)
end
