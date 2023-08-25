using JLD2, FileIO

abstract type AbstractTest end
struct Test
    a::Any
end

g(t::Test) = t.a*2

save("test.jld2", "test", Test(1))