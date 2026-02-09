using BenchmarkTools
const selector = (1,5,10)
const selectorbig = (1,10,50,100,1000,1010,2000)

const selectorT = Type{selector}
const selectorbigT = Type{selectorbig}

struct Foo{T}
    a::T
end

struct Foos
    v::Vector{Foo}
end

const f1 = Foo(10)
const f2 = Foo(100.)
const f3 = Foo(1000)
const f4 = Foo(10000.)
const f5 = Foo(100)
const f6 = Foo(1000.)
const fs = Foos([f1,f2,f3,f4,f5,f6])

function choice(ts1, ts2, ::Type{Type{T}}) where T
    # println(T)
    idx = rand(T[1][1]:T[2][end])
    if idx <= T[1][end]
        return choicefunc(ts1, idx)
    else
        return choicefunc(ts2, idx)
    end
end

@generated function gchoice(fs, ::Type{Type{Partitions}}) where Partitions
    len = length(Partitions)
    startidx = Partitions[1]
    endidx = Partitions[end]


    str = "begin
        idx = rand($startidx:$endidx)
        "
    for part_idx in 2:(len-1)
        str *= "if idx <= $(Partitions[part_idx])
            return choicefunc(fs.v[$(part_idx-1)], idx)
        else"
    end

    str *= "
    return choicefunc(fs.v[$(len-1)], idx)
    end
    end"

    return Meta.parse(str)
end

function choice(fs, partitions)
    idx = rand(partitions[1]:partitions[end])
    for it_idx in 2:(length(partitions)-1)
        if idx <= partitions[it_idx]
            return choicefunc(fs.v[it_idx-1], idx)
        end
    end
    return choicefunc(fs.v[length(partitions)-1], idx)
end

function choice(fs, partitions::Type{Type{Partitions}}) where Partitions
    idx = rand(Partitions[1]:Partitions[end])
    for it_idx in 2:(length(Partitions)-1)
        if idx <= Partitions[it_idx]
            return choicefunc(fs.v[it_idx-1], idx)
        end
    end
    return choicefunc(fs.v[length(Partitions)-1], idx)
end

function choicefunc(ts::Foo{Int}, idx)
    return Float64(ts.a^5 + idx)
end

function choicefunc(ts::Foo{Float64}, idx)
    return ts.a^5 + 0.4 + Float64(idx)
end

@time for _ in 1:10^8
    choice(fs, selectorbig)
end

@time for _ in 1:10^8
    choice(fs, selectorbigT)
end

@time for _ in 1:10^8
    gchoice(fs, selectorbigT)
end