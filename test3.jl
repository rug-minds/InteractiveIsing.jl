function reshapeView(vec, start, length, width)
    return reshape((@view vec[start:(start + width*length - 1)]), length, width)
end

struct A
      vec
end

struct B
      mat
end

a = A(rand(100))

b = B(reshapeView(a.vec, 3, 3, 3))

struct C{T,N}
    arr::Array{T,N}
end

C(vec::Vector{T}) where T = C{T,2}(reshapeView(vec, 3, 3, 3))

c = C(a.vec)