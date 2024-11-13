struct UPair
    x::Int64
    y::Int64
end

function Base.show(io::IO, p::UPair)
    print(io, "$(min(p.x,p.y)) <=> $(max(p.x,p.y))")
end

#extend ==
function Base.:(==)(a::UndirectedPair, b::UndirectedPair)
    return (a.x == b.x && a.y == b.y) || (a.x == b.y && a.y == b.x)
end

# Custom hash
function Base.hash(a::UndirectedPair)
    return UInt((2 << a.x) + (2 << a.y))
end

⇔(i1, i2) = UPair(i1, i2)
export ⇔