using BenchmarkTools
function testmixed1()
    num = 0.
    for i in 1:10000
        num += rand(Int32[1,2])*rand(Float64)
    end
    return num
end
function testmixed2()
    num = 0.
    for i in 1:10000
        num += rand(Int64[1,2])*rand(Float32)
    end
    return num
end
function test32()
    num = 0.f0
    for i in 1:10000
        num += rand(Int32[1,2])*rand(Float32)
    end
    return num
end
function test64()
    num = 0.
    for i in 1:10000
        num += rand(Int64[1,2])*rand(Float64)
    end
    return num
end
@benchmark testmixed1()
@benchmark testmixed2()
@benchmark test32()
@benchmark test64()