# Analysis functions
export dataToDF, tempSweep, MPlot, sampleCorrPeriodic, sampleCorrPeriodicDefects, corrFuncXY, dfMPlot, dataFolderNow, csvToDF, corrPlotDF, corrPlotDFSlices, corrPlotTRange

mutable struct AInt32
    @atomic x::Int32
end

mutable struct AFloat32
    @atomic x::Float32
end

# Data structure for gathering averages and gauging convergence
include("Plotting.jl")
include("CorrelationLength.jl")
include("Data.jl")
include("TempSweep.jl")
include("TotalEnergy.jl")
include("Susceptibility.jl")
include("User.jl")
