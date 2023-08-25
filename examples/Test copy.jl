using JLD2, FileIO

struct Test2v2
    a::Any
    b::Any
end

t = load("test.jld2", "test")