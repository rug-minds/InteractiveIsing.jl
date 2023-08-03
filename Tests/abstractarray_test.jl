struct WrappedInt
    i::Int64
end
    
struct Bar
    i::WrappedInt
end

struct Foo
    b::Vector{Bar}
end

struct WIntArray <: AbstractArray end

function Base.iterate(wi::WI)


#make iterate function for WIntArray


