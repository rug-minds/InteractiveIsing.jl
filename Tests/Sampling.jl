using Distributions, BenchmarkTools

mutable struct SampleRangeVal
    rangebegin::Float32
    rangelength::Float32
end

mutable struct SampleRangeDistribution
    d::Uniform{Float32}
end

mutable struct SampleRangeType{T} end

mutable struct SampleRangeRange
    r::UnitRange{Float32}
end

srv = SampleRangeVal(-1f0, 3f0)
srd = SampleRangeDistribution(Uniform(-1f0, 2f0))
srt = SampleRangeType{(-1f0, 2f0)}()
srr = SampleRangeRange(UnitRange(-1f0,2f0))

function samplerangeval(srv)
    cum = 0f0
    for _ in 1:10000
        cum += rand(Float32)*srv.rangelength + srv.rangebegin
    end
    return cum
end

function samplerangedist(srd)
    cum = 0f0
    for _ in 1:10000
        cum += Float32(rand(srd.d))
    end
    return cum
end

function samplerangetype(srt::SampleRangeType{T}) where T
    dist = Uniform(T[1], T[2])
    cum = 0f0
    for _ in 1:10000
        cum += Float32(rand(dist))
    end
    return cum
end

function samplerangerange(srr::SampleRangeRange)
    cum = 0f0
    for _ in 1:10000
        cum += rand(Float32)*Float32(length(srr.r)) + srr.r.start
    end
    return cum
end

@benchmark samplerangeval($srv)
@benchmark samplerangedist($srd)
# @benchmark samplerangetype($srt)
# @benchmark samplerangerange($srr)

# samplerangeval(srv)
# samplerangedist(srd)
