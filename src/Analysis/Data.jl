
mutable struct DataPoint
    M::     Union{Nothing, Float32}
    bins::  Union{Nothing, Vector{Float32}}
    corrs:: Union{Nothing, Vector{Float32}}
    img::   Union{Nothing, Array}
    other:: Dict{Symbol, Any}
    
    DataPoint(M,bins,corrs) = new(M,bins,corrs,nothing, Dict{Symbol, Any}())
end

DataPoint() = DataPoint(0.0, Float32[], Float32[])

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
function Base.show(io::IO, d::DataPoint)
    print("Data point with")
    hasM = !isnothing(d.M)
    hascorrs = !isnothing(d.corrs)
    hasimg = !isnothing(d.img)
    hasother = !isempty(d.other)

    hasM && print("\n\tMagnetization: $(d.M)")
    hascorrs && print("\n\tCorrelation data")
    hasimg && print("\n\tImage data")
    if hasother 
        println("\n\tOther data")
        print(keys(d.other))
    end
    if !hasM && !hascorrs && !hasimg && !hasother
        println(" no data")
    end
end
Base.display(d::DataPoint) = show(d)
# @ForwardDeclare LayerDataPoints
struct TempDataPoints{DP <: DataPoint} <: AbstractVector{DP}
    data::Vector{DP}
    temp::Float32
end

function Base.show(io::IO, tdp::TempDataPoints)
    println(io, "Data for temperature $(tdp.temp)")
    multiple = length(tdp.data) > 1
    println(io, "$(length(tdp.data)) recorded datapoint"*("s"^(multiple ? 1 : 0)))
end
Base.display(tdp::TempDataPoints) = show(tdp)

TempDataPoints(ndatapoints::Integer, temp::Real) = TempDataPoints(DataPoint[DataPoint() for _ in 1:ndatapoints], Float32(temp))

size(tdp::TempDataPoints) = size(tdp.data)
getindex(tdp::TempDataPoints, idx::Integer) = tdp.data[idx]
getindex(tdp::TempDataPoints, idxs::Integer...) = tdp.data[idxs...]
setindex!(tdp::TempDataPoints, value, idx::Integer) = tdp.data[idx] = value

magnetization(tempdata::TempDataPoints) = sum([point.M for point in tempdata])/length(tempdata)


struct LayerDataPoints{TDP <: TempDataPoints} <: AbstractVector{TDP}
    data::Vector{TDP}
    layeridx::Int32
end

function Base.show(io::IO, ldp::LayerDataPoints{TempDataPoints})
    println(io, "Data for layer $(ldp.layeridx):")
    println(io, "$(length(ldp.data)) recorded temperatures: ")
    print("| ")
    for idx in 1:length(ldp.data)    
        print(io, ldp.data[idx].temp, " | ")
    end
end

Base.display(ldp::LayerDataPoints) = show(ldp)

LayerDataPoints(temps, ndatapoints, lidx) = LayerDataPoints(TempDataPoints[TempDataPoints(ndatapoints, temp) for temp in temps], Int32(lidx))

size(ldp::LayerDataPoints) = size(ldp.data)
getindex(ldp::LayerDataPoints, idx::Integer) = ldp.data[idx]
getindex(ldp::LayerDataPoints, idxs::Integer...) = ldp.data[idxs...]
setindex!(ldp::LayerDataPoints, value, idx::Integer) = ldp.data[idx] = value

struct TempSweepData <: AbstractDict{String, LayerDataPoints}
    # Entry for each layer
    # Then entry for each temperature
    # Then an entry for each datapoint
    layers::Dict{String, LayerDataPoints}
end
TempSweepData() = TempSweepData(Dict{String, LayerDataPoints}())

getindex(tsd::TempSweepData, key::String) = getindex(tsd.layers, key)
getindex(tsd::TempSweepData, key::Integer) = getindex(tsd.layers, string(key))
setindex!(tsd::TempSweepData, value, key::String) = setindex!(tsd.layers, value, key)
Base.keys(tsd::TempSweepData) = keys(tsd.layers)
Base.values(tsd::TempSweepData) = values(tsd.layers)
Base.iterate(tsd::TempSweepData, state = 1) = iterate(tsd.layers, state)
Base.length(tsd::TempSweepData) = length(tsd.layers)

function TempSweepData(g, layeridxs, temps, ndatapoints; original_lidx = false)
    # nlayers = length(layeridxs)
    layerdata = TempSweepData()
    for (i, lidx) in enumerate(layeridxs)
        idx = original_lidx ? lidx : i
        layerdata[string(idx)] = LayerDataPoints(temps, ndatapoints, lidx)
    end
    return layerdata
end

function Base.show(io::IO, tsd::TempSweepData)
    ks = keys(tsd)
    print(io, "TempSweepData with $(length(ks)) recorded layers: $ks")
end

Base.display(tsd::TempSweepData) = show(tsd)

layeridxs(tsd::TempSweepData) = [internal_idx(layer) for layer in tsd.layers]

function magnetizationPlot(tsData::TempSweepData, layer::Integer)
    ldata = tsData.layers[layer]
    temps = [tempdata.temp for tempdata in ldata]

    ms = [magnetization(tempdata) for tempdata in ldata]

    plot(temps, ms, label = "M = $M")
end

"""
Create folder in data folder with given name and return path
Tries to create a folder if it doesn't exist, and doesn't do anything if it does
"""
function dataFolderNow(subfoldername::String; foldername::String = "Data", startfolder = nothing)
    nowtime = getnowtime()
    startfolder = isnothing(startfolder) ? dirname(Base.source_path()) : startfolder
    try; mkdir(joinpath(startfolder, foldername)); catch end
    try; mkdir(joinpath(startfolder, foldername, subfoldername)); catch end
    try; mkdir(joinpath(startfolder, foldername, subfoldername, "$nowtime")); catch end
    return "$foldername/$subfoldername/$nowtime/"
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