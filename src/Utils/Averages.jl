"""
Struct to gather averages of arbitrary functions and gauge convergence
"""
mutable struct AvgData{Func<:Function}
    const data::Vector{Float64}
    const avgs::Vector{Float64}
    const windowavgs::Vector{Float64}
    const func::Func
    lastsum::Float64
    windowsize::Int
    convergence::Float64
    converged::Bool
    convergence_step::Int
end
export AvgData
"""
Function is the function to average over
windowsize is the size of the window to gauge convergence
Storagesize is the initial storage size for faster pushing
convergence is the convergence threshold of the window average 
"""
function AvgData(func::Function = identity; windowsize::Int = 4, storagesize = 128, convergence = 1e-5) 
    data = Float64[]
    avgs = Float64[]
    sizehint!(data, storagesize)
    sizehint!(avgs, storagesize)
    windowavgs = Float64[]
    sizehint!(windowavgs, storagesize - windowsize + 1)

    return AvgData(data, avgs, windowavgs, func, 0, windowsize, convergence, false, 0)
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
plot_conv(ad) = pl.plot(ad.windowavgs, xlabel = "Step", ylabel = "Window RMSD", title = "Convergence")
export plot_conv
plot_avgs(ad) = pl.plot(ad.avgs, xlabel = "Step", ylabel = "Average", title = "Averages")
export plot_avgs

# avg(vec) = sum(vec)/length(vec)
avg(ad::AvgData) = ad.lastsum/length(ad.data)


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

Base.push!(ad::AvgData, x) = begin
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