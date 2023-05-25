using BenchmarkTools
const selector = Type{(1:5,6:10)}
const selector2 = Type{(1:3,4:10)}
const selectorbig = Type{(1:999999,1000000:100000000)}
struct TestStruct{T}
    a::T
end

const ts1 = TestStruct(10)
const ts2 = TestStruct(100.)

function choice(ts1, ts2, ::Type{Type{T}}) where T
    # println(T)
    idx = rand(T[1][1]:T[2][end])
    if idx <= T[1][end]
        return choicefunc(ts1, idx)
    else
        return choicefunc(ts2, idx)
    end
end

@generated function gchoice(ts1, ts2, ::Type{Type{T}}) where T
    return Meta.parse("begin
        idx = rand(T[1][1]:T[2][end])
        if idx <= T[1][end]
            return choicefunc(ts1, idx)
        else
            return choicefunc(ts2, idx)
        end
    end")
end
@generated function gchoice2(ts1, ts2, ::Type{Type{T}}) where T
    rangestart = T[1][1]
    rangeend = T[2][end]
    split = T[1][end]
    return Meta.parse("begin
        idx = rand($rangestart:$rangeend)
        if idx <= $split
            return choicefunc(ts1, idx)
        else
            return choicefunc(ts2, idx)
        end
    end")
end

const it1 = 1:5
const it2 = 6:10
const it1big = 1:999999
const it2big = 1000000:100000000
function choice(ts1, ts2, it1::UnitRange, it2::UnitRange)
    idx = rand(it1[1]:it2[2])
    if idx <= it1[end]
        return choicefunc(ts1, idx)
    else
        return choicefunc(ts2, idx)
    end
end

function choicefunc(ts::TestStruct{Int}, idx)
    return Float64(ts.a^5 + idx)
end

function choicefunc(ts::TestStruct{Float64}, idx)
    return ts.a^5 + 0.4 + Float64(idx)
end