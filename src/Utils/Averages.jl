"""
Struct to gather averages of arbitrary functions and gauge convergence
"""
mutable struct AvgData{T, Func<:Function}
    const data::Vector{T}
    const avgs::Vector{T}
    const windowavgs::Vector{T}
    const func::Func
    lastsum::T
    windowsize::Int
    convergence::T
    converged::Bool
    convergence_step::Int
end
export AvgData
"""
Function is the function to average over
windowsize is the size of the window to gauge convergence
Storagesize is the initial storage size for faster pushing
convergence is the convergence threshold of the window average 

AvgData(func::Function = identity; windowsize::Int = 4, storagesize = 128, convergence = 1e-5) 
"""
function AvgData(T, func = identity; windowsize::Int = 4, storagesize = 128, convergence = 1e-5)
    data = T[]
    avgs = T[]
    sizehint!(data, storagesize)
    sizehint!(avgs, storagesize)
    windowavgs = T[]
    sizehint!(windowavgs, storagesize - windowsize + 1)

    return AvgData{T, typeof(func)}(data, avgs, windowavgs, func, T(0), windowsize, T(convergence), false, 0)
end

function reset!(ad::AvgData{T}, sz = 128) where T
    deleteat!(ad.data, 1:length(ad.data))
    deleteat!(ad.avgs, 1:length(ad.avgs))
    deleteat!(ad.windowavgs, 1:length(ad.windowavgs))

    sizehint!(ad.data, sz)
    sizehint!(ad.avgs, sz)
    ad.lastsum = 0
    ad.converged = false
    ad.convergence_step = 0
    return ad
end

converged(ad::AvgData) = ad.converged
export converged
Base.isempty(ad::AvgData) = isempty(ad.data)
Base.length(ad::AvgData) = length(ad.data)
Base.sizehint!(ad::AvgData, n::Int) = begin
    sizehint!(ad.data, n)
    sizehint!(ad.avgs, n)
    sizehint!(ad.windowavgs, n - ad.windowsize + 1)
    return ad
end
Base.getindex(ad::AvgData) = ad.avgs[end]
# plot_conv(ad) = pl.plot(ad.windowavgs, xlabel = "Step", ylabel = "Window RMSD", title = "Convergence")
export plot_conv
# plot_avgs(ad) = pl.plot(ad.avgs, xlabel = "Step", ylabel = "Average", title = "Averages")
export plot_avgs

# avg(vec) = sum(vec)/length(vec)
avg(ad::AvgData) = length(ad.data) != 0 ? ad.lastsum/length(ad.data) : 0
avg(sub::SubArray) = length(sub) != 0 ? sum(sub)/length(sub) : 0

function window_rmsd(ad::AvgData)
    avgs = ad.avgs
    w_size = ad.windowsize
    if length(avgs) < ad.windowsize
        return nothing
    end
    data = @view avgs[end-w_size+1:end]
    window_avg = avg(data)
    sq = 0.
    for avg in data
        sq += (avg - window_avg)^2
    end
    return sqrt(sq/w_size)
end
export window_rmsd

function Base.push!(ad::AvgData{T}, x) where {T}
    push!(ad.data, ad.func(x))
    ad.lastsum += ad.data[end]
    push!(ad.avgs, avg(ad))
    w_rmsd = window_rmsd(ad)
    if !isnothing(w_rmsd)
        push!(ad.windowavgs, w_rmsd)
        if w_rmsd < ad.convergence
            ad.converged = true
            if ad.convergence_step == 0
                ad.convergence_step = length(ad.data)
            end
        end
    end
    return ad
end


mutable struct FunctionAverage{T,F}
    data::Vector{T}
    f::F
    sum::T
end

function FunctionAverage(T, f::Function)
    return FunctionAverage{T, typeof(f)}(T[], f, T(0))
end

function Base.push!(fa::FunctionAverage, x)
    datum = fa.f(x)
    push!(fa.data, datum)
    fa.sum += datum
    return fa
end

Base.length(fa::FunctionAverage) = length(fa.data)
Base.size(fa::FunctionAverage) = size(fa.data)

function reset!(fa::FunctionAverage)
    deleteat!(fa.data, 1:length(fa.data))
    fa.sum = 0
    return fa
end

avg(fa::FunctionAverage) = fa.sum/length(fa.data)

mutable struct STDev{T}
    d::Vector{T}
    sum::T
    sumsq::T
end

function STDev(T)
    return STDev{T}(T[], T(0), T(0))
end
export STDev

Base.push!(sd::STDev, x) = begin
    push!(sd.d, x)
    sd.sum += x
    sd.sumsq += x^2
    return sd
end

Base.length(sd::STDev) = length(sd.d)
Base.size(sd::STDev) = size(sd.d)
Base.get(sd::STDev) = sd.sumsq/length(sd.d) - (sd.sum/length(sd.d))^2
# getsd(sd::STDev) = sd.sumsq/length(sd.d) - (sd.sum/length(sd.d))^2
Base.iterate(sd::STDev, state = 1) = iterate(sd.d, state)
reset!(sd::STDev) = begin
    deleteat!(sd.d, 1:length(sd.d))
    sd.sum = 0
    sd.sumsq = 0
    return sd
end