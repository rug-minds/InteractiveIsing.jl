mutable struct DataPoint <: AbstractDict{Symbol, Any}
    M::     Float32
    bins::  Vector{Float32}
    corrs:: Vector{Float32}
    img::Union{Nothing, Array}
    other:: Dict{Symbol, Any}
    DataPoint(M,bins,corrs) = new(M,bins,corrs,nothing,Dict{Symbol, Any}())
end

DataPoint() = DataPoint(0.0, Float32[], Float32[], Dict{Symbol, Any}())

function getindex(d::DataPoint, key::Symbol)
    if key == :M
        return d.M
    elseif key == :bins
        return d.bins
    elseif key == :corrs
        return d.corrs
    else
        return d.other[key]
    end
end

function setindex!(d::DataPoint, value, key::Symbol)
    if key == :M
        return d.M = value
    elseif key == :bins
        return d.bins = value
    elseif key == :corrs
        return d.corrs = value
    else
        return d.other[key] = value
    end
end

Base.keys(d::DataPoint) = [:M, :bins, :corrs, keys(d.other)...]

struct TempDataPoints <: AbstractVector{DataPoint}
    data::Vector{DataPoint}
    temp::Float32
end
size(tdp::TempDataPoints) = size(tdp.data)
getindex(tdp::TempDataPoints, idx::Integer) = tdp.data[idx]
getindex(tdp::TempDataPoints, idxs::Integer...) = tdp.data[idxs...]
setindex!(tdp::TempDataPoints, value, idx::Integer) = tdp.data[idx] = value

magnetization(tempdata::TempDataPoints) = sum([point.M for point in tempdata])/length(tempdata)


struct LayerDataPoints <: AbstractVector{TempDataPoints}
    data::Vector{TempDataPoints}
    layeridx::Int32
end

size(ldp::LayerDataPoints) = size(ldp.data)
getindex(ldp::LayerDataPoints, idx::Integer) = ldp.data[idx]
getindex(ldp::LayerDataPoints, idxs::Integer...) = ldp.data[idxs...]
setindex!(ldp::LayerDataPoints, value, idx::Integer) = ldp.data[idx] = value

struct TempSweepData <: AbstractDict{String, Any}
    # Entry for each layer
    # Then entry for each temperature
    # Then an entry for each datapoint
    layers::Dict{String, LayerDataPoints}
end

getindex!(tsd::TempSweepData, key::String) = getindex(tsd.layers, key)
getindex!(tsd::TempSweepData, key::Integer) = getindex(tsd.layers, String(key))
setindex!(tsd::TempSweepData, value, key::String) = setindex!(tsd.layers, value, key)
Base.keys(tsd::TempSweepData) = keys(tsd.layers)
Base.values(tsd::TempSweepData) = values(tsd.layers)
Base.iterate(tsd::TempSweepData, state = 1) = iterate(tsd.layers, state)

function TempSweepData(g, layeridxs, temps, ndatapoints; original_lidx = false)
    nlayers = length(layeridxs)
    layers = TempSweepData(Dict{LayerDataPoints}())
    for (i, lidx) in enumerate(layeridxs)
        idx = original_lidx ? lidx : i
        layers[String(idx)] = LayerDataPoints([TempDataPoints([DataPoint() for _ in 1:ndatapoints], t) for t in temps], lidx)
    end
    return layers
end

layeridxs(tsd::TempSweepData) = [layer.layeridx for layer in tsd.layers]

function magnetizationPlot(tsData::TempSweepData, layer::Integer)
    ldata = tsData.layers[layer]
    temps = [tempdata.temp for tempdata in ldata]

    ms = [magnetization(tempdata) for tempdata in ldata]

    plot(temps, ms, label = "M = $M")
end

savaSweepData(tsd::TempSweepData; foldername::String = dataFolderNow("TempSweepData")) = save(foldername, tsd)
openSweepData(filename) = load(filename)

"""
Save magnetization data to a dataframe
"""
function MToDF(M, dpoint::Integer = Int32(1), T::Real = Float16(1))
    return DataFrame(M = M, D = dpoint, T = T)
end

""" 
Determine amount of datapoints per temperature from dataframe data (if homogeneous over temps)
"""
function detDPoints(dat)
    dpoint = 0
    lastt = dat[:,1][1]
    for temp in dat[:,1]
        if temp == lastt
            dpoint += 1
        else 
            return dpoint
        end
    end
    return dpoint
end

"""
Input the temperature sweep data into a dataframe
"""
function dataToDF(tsData, lMax = length(tsData[2][1]))
    return DataFrame(Temperature = tsData[1], Correlation_Function = [corrLXY[1:lMax] for corrLXY in  tsData[2]], Magnetization = tsData[3] )
end

"""
Read CSV and outputs dataframe
Make this use julia data format
"""
csvToDF(filename) = DataFrame(CSV.File(filename)) 

"""
Helper Functions 
"""
function insertShift(vec::Vector{T}, el) where T
    newVec = Vector{T}(undef, length(vec))
    newVec[1:(end-1)] = @view vec[2:end]
    newVec[end] = T(el)
    return newVec
end

""" 
Create folder in data folder with given name and return path
Tries to create a folder if it doesn't exist, and doesn't do anything if it does
"""
function dataFolderNow(subfoldername::String, foldername::String = "Data")
    nowtime = "$(now())"[1:(end-7)]
    try; mkdir(joinpath(dirname(Base.source_path()), foldername)); catch end
    try; mkdir(joinpath(dirname(Base.source_path()), foldername, subfoldername)); catch end
    try; mkdir(joinpath(dirname(Base.source_path()), foldername, subfoldername, "$nowtime")); catch end
    return "Data/Tempsweep/$nowtime/"
end

"""
Try to make a folder and don't do anything if it's already there
"""
function trymakefolder(foldername::String)
    if foldername[end] == '/'
        foldername = foldername[1:(end-1)]
    end
    try; mkdir(joinpath(dirname(Base.source_path()),foldername)); catch end
end