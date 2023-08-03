# Analysis functions
export dataToDF, tempSweep, MPlot, sampleCorrPeriodic, sampleCorrPeriodicDefects, corrFuncXY, dfMPlot, dataFolderNow, csvToDF, corrPlotDF, corrPlotDFSlices, corrPlotTRange

mutable struct AInt32
    @atomic x::Int32
end

mutable struct AFloat32
    @atomic x::Float32
end

include("Plotting.jl")
include("CorrelationLength.jl")
include("Data.jl")
include("TempSweep.jl")
include("User.jl")


# Correlation Length Data
# Save correlation date (lVec,corrVec) to dataframe
function corrToDF((lVec,corrVec), dpoint::Integer = Int32(1), T::Real = Float16(1))
    return DataFrame(L = lVec, Corr = corrVec, D = dpoint, T = T)
end

# Not used currently, used to fit correleation length data to a function f
function fitCorrl(dat,dom_end, f, params...)
    dom = Domain(1.:dom_end)
    data = Measures(Vector{Float64}(dat[1:dom_end]),0.)
    model = Model(:comp1 => FuncWrap(f,params...))
    prepare!(model,dom, :comp1)
    return fit!(model,data)
end
