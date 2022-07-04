# Analysis functions

__precompile__()

module Analysis

using FileIO, Images, LaTeXStrings, Dates, DataFrames, CSV
import Plots as pl
# include("IsingGraphs.jl")
using ..IsingGraphs

include("SquareAdj.jl")
using .SquareAdj

export magnetization, dataToDF, tempSweep, MPlot, sampleCorrPeriodic, corrFuncXY, dfMPlot, dataFolderNow

mutable struct AInt32
    @atomic x::Int32
end

mutable struct AFloat32
    @atomic x::Float32
end


"""
User functions
"""

# Does analysis over multiple temperatures.
# Analyzes magnetization and correlation length.
# Usage: Goes from current temperature of simulation, to TF, in stepsizes of TStep.
#   Every step, a number of datapoints (dpoints) are recorded, with inervals between them of dpointwait
#   In between Temperatures there is an optional wait time of stepwait, for equilibration
#   There is an initial wait time of equiwait for equilibration
function tempSweep(g, TIs, M_array; TI = TIs[], TF = 13, TStep = 0.5, dpoints = 6, dpointwait = 5, stepwait = 0, equiwait = 0, saveImg = false, img = Ref([]), corrF = sampleCorrPeriodic)
    """ Make folder """
    foldername = dataFolderNow("Tempsweep")

    println("Starting temperature sweep")
    println("Parameters TIs$TIs, M_array$M_array, TI$TI,TF$TF, TStep$TStep, dpoints$dpoints, dpointwait$dpointwait, stepwait$stepwait, equiwait$equiwait, saveImg$saveImg")
    
    first = true
    TRange =  TI:TStep:TF
    
    # DataFrames
    mdf = DataFrame()
    cldf = DataFrame()
    for T in TRange
        println("Gathering data for temperature $T, $dpoints data points in intervals of $dpointwait seconds")
        waittime = length(T:TStep:TF)*(dpoints*dpointwait + stepwait)
        if first # If first run, wait equiwait seconds for equibrilation
            println("The sweep will take approximately $(equiwait + waittime) seconds")
            if equiwait != 0
                println("Waiting $equiwait seconds for equilibration")
                sleep(equiwait)
            end
            first=false
        else # Else wait in between temperature steps
            println("Now waiting $stepwait seconds in between temperatures")
            sleep(stepwait)
        end

        # User feedback of remaining time
        println("Approximately $(length(waittime)) seconds remaining")
        
        TIs[] = T

        # Gather Datapoints
        for point in 1:dpoints
            tpointi = time()
            (lVec,corrVec) = corrF(g) 
            cldf = vcat(cldf, corrToDF((lVec,corrVec) , point, TIs[]) )
            mdf = vcat(mdf,MToDF(last(M_array[]),point, TIs[]) )
            
            if saveImg
                # Image of ising
                save(File{format"PNG"}("$(foldername)Ising T$T d$point.PNG"), img[])

                # Image of correlation plot
                corrPlot = pl.plot(lVec,corrVec, xlabel = "Length", ylabel=L"C(L)")
                Tstring = replace("$T", '.' => ',')
                pl.savefig(corrPlot,"$(foldername)Corrplot T$Tstring $point")
            end

            tpointf = time()
            dtpoint = tpointf-tpointi

            println("Datapoint took $dtpoint seconds")

            sleep(max(dpointwait-dtpoint,0))
        end
        
    end

    if saveImg
        dfMPlot(mdf, foldername)
    end

    CSV.write("$(foldername)CorrData", cldf)
    CSV.write("$(foldername)MData", mdf)
    
end



""" Correlation length calculations """

rthetas = 2*pi .* rand(10^7) # Saves random angles to save computation time

# Takes a random number of pairs for every length to calculate correlation length function
# This only works well with defects.
function sampleCorrPeriodic(g::IsingGraph, Lstep::Float16 = Float16(.5), lStart::Integer = Int8(1), lEnd::Integer = Int16(256), npairs::Integer = Int16(10000) )
    function sigToIJ(sig, L)
        return (L*cos(sig),L*sin(sig))
    end

    function sampleIdx2(idx1,L,rtheta)
        ij = idxToCoord(idx1,g.N)
        dij = Int32.(round.(sigToIJ(rtheta,L)))
        idx2 = coordToIdx(latmod.((ij.+dij),g.N),g.N)
        return idx2
    end

    theta_i = rand([1:length(rthetas);])

    avgsum = (sum(g.state)/g.size)^2

    lVec = [lStart:Lstep:lEnd;]
    corrVec = Vector{Float32}(undef,length(lVec))

    # Sample all idx to be used
    # Slight bit faster to do it this way than to sample it every time in the loop
    idx1s = rand(g.aliveList,length(lVec)*npairs)
    # Index of above vector
    idx1idx = 1
    # Iterate over all lengths to be checked
    for (lidx,L) in enumerate(lVec)
 
        sumprod = 0 #Track the sum of products sig_i*sig_j
        for _ in 1:npairs
            idx1 = idx1s[idx1idx]
            rtheta = rthetas[(theta_i -1) % length(rthetas)+1]
            idx2 = sampleIdx2(idx1,L,rtheta)
            sumprod += g.state[idx1]*g.state[idx2]
            theta_i += 1 # Sample next random angle
            idx1idx += 1 # Sample next random idx
        end
        # println((sum(g.state)/g.N)^2)
        # println(avgsum1*avgsum2/(npairs^2))
        corrVec[lidx] = sumprod/npairs - avgsum
    end

    return (lVec,corrVec)
end

"""Correlation Length Data"""
function corrToDF((lVec,corrVec), dpoint::Integer = Int8(1), T::Real = Float16(1))
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


""" Magnetization """
function MToDF(M, dpoint::Integer = Int8(1), T::Real = Float16(1))
    return DataFrame(M = M, D = dpoint, T = T)
end

function dfMPlot(mdf::DataFrame, foldername::String)
    """ DF FORMAT """
    """ M : D : T """

    # All magnetizations
    ms = mdf[:,1]
    # All Temperatures
    ts = mdf[:,3]
    slices = []

    lastt = ts[1]
    lasti = 1
    for idx in 2:length(ts)
        if !(ts[idx] == lastt)
            append!(slices,[lasti:(idx-1)])
            lasti=idx
        end
        lastt = ts[idx]
    end
    append!(slices, [lasti:length(ts)])

    avgM = Vector{Float32}(undef,length(slices))
    newTs = Vector{Float32}(undef,length(slices))
    for (idx,slice) in enumerate(slices)
        avgM[idx] = sum(ms[slice])/length(slice)
        newTs[idx] = ts[first(slice)]
    end

    if avgM[1] < 0
        avgM .*= -1
    end
    
    mPlot = pl.plot(newTs,avgM, xlabel = "T", ylabel="M", label=false)
    pl.savefig(mPlot,"$(foldername)Mplot")
end

# Averages M_array over an amount of steps
# Updates magnetization (which is thus the averaged value)
let avg_window = 60, frames = 0
    global function magnetization(g::IsingGraphs.IsingGraph,M, M_array)
        avg_window = 60 # Averaging window = Sec * FPS, becomes max length of vector
        M_array[] = insertShift(M_array[], Int32(sum(g.state)))
        if frames > avg_window
            M[] = sum(M_array[])/avg_window 
            frames = 0
        end 
        frames += 1 
    end
end

"""General data"""

# Determine amount of datapoints per temperature from dataframe data (if homogeneous over temps)
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


# Input the temperature sweep data into a dataframe
function dataToDF(tsData, lMax = length(tsData[2][1]))
    return DataFrame(Temperature = tsData[1], Correlation_Function = [corrLXY[1:lMax] for corrLXY in  tsData[2]], Magnetization = tsData[3] )
end

# Read CSV and outputs dataframe
csvToDF(filename) = DataFrame(CSV.File(filename)) 

""" Helper Functions"""
function insertShift(vec::Vector{T}, el::T) where T
    newVec = Vector{T}(undef, length(vec))
    newVec[1:(end-1)] = vec[2:end]
    newVec[end] = el
    return newVec
end

# Create folder in data folder with given name and return path
function dataFolderNow(dataString::String)
    nowtime = "$(now())"[1:(end-7)]
    try; mkdir(joinpath(dirname(Base.source_path()),"Data")); catch end
    try; mkdir(joinpath(dirname(Base.source_path()), "Data", dataString)); catch end
    try; mkdir(joinpath(dirname(Base.source_path()), "Data", dataString, "$nowtime")); catch end
    return "Data/Tempsweep/$nowtime/"
end



""" OLD STUFF """

# Sample random spins for correlation length, but make sure every pair is only sampled once
# Colissions are quite unlikely for smaller number of sampled pairs, making this redundant and slower
function sampleCorrPeriodicUnique(g::IsingGraph, Lstep::Float16, lStart::Int8 = 1, lEnd::Int16 = 256, npairs::Integer = 1000 )
    function sigToIJ(sig, L)
        return (L*cos(sig),L*sin(sig))
    end

    function sampleIdx2(idx1,L,rtheta)
        ij = idxToCoord(idx1,g.N)
        dij = Int32.(round.(sigToIJ(rtheta,L)))
        idx2 = coordToIdx(latmod.((ij.+dij),g.N),g.N)
        return idx2
    end

    theta_i = rand([1:length(rthetas);])

    avgsum = (sum(g.state)/g.size)^2

    lVec = [lStart:Lstep:lEnd;]
    corrVec = Vector{Float32}(undef,length(lVec))

    pairs =  Set() # To check wether pair is already checked

    # Iterate over all lengths to be checked
    for (lidx,L) in enumerate(lVec)
        pairs_done = 0
 
        sumprod = 0 #Track the sum of products sig_i*sig_j
        while pairs_done <= npairs
            idx1 = rand(g.aliveList)
            rtheta = rthetas[(theta_i -1) % length(rthetas)+1]
            idx2 = sampleIdx2(idx1,L,rtheta)
            
            if !((idx1,idx2) in pairs)
                sumprod += g.state[idx1]*g.state[idx2]
                pairs_done +=1
                union!(pairs,(idx1,idx2))
            else
                continue
            end
            theta_i += 1 #sample next random angle
        end
        # println((sum(g.state)/g.N)^2)
        # println(avgsum1*avgsum2/(npairs^2))
        corrVec[lidx] = sumprod/npairs - avgsum
    end

    return (lVec,corrVec)

end

# Sweep the lattice to find x and y correlation data.
function corrLXY(g::IsingGraphs.IsingGraph, L)
    avgprod = 0
    prodavg1 = 0
    prodavg2 = 0
    Mprod = 0
    M1 = 0
    M2 = 0
    # filter = [ #only do for spin pairs within matrix
    #     let (i1,j1) = idxToCoord(state,g.N), i2 = i1+L, j2 = j1+L
    #         i2 <= g.N && j2 <=g.N
    #     end
    #     for state in 1:g.size
    # ]
    # for state1 in g.state[filter]
    for stateIdx in g.aliveList

        state1 = g.state[stateIdx]
        (i1,j1) = idxToCoord(stateIdx,g.N)
        i2 = i1+L
        j2 = j1+L

        # Check if points are added
        addedi2 = false
        addedj2 = false

        if i2 < g.N && !g.defectBools[coordToIdx(i2,j1,g.N)]
            addedi2 = true

            statey = g.state[coordToIdx(i2,j1,g.N)]
            prodavg2 += statey   
            avgprod += state1*statey
            Mprod += 1
            M2 +=1
        end
        
        if j2 < g.N && !g.defectBools[coordToIdx(i1,j2,g.N)]
            addedj2 = true
            statex = g.state[coordToIdx(i1,j2,g.N)]
            prodavg2 += statex
            avgprod += state1*statex
            Mprod += 1
            M2 +=1
        end

        if addedi2 || addedj2
            prodavg1 += state1 
            M1 += 1
        end
    end

    return avgprod/Mprod-prodavg1*prodavg2/(M1*M2)

end

# Calculates the two points correlation function for different lengths and returns a vector with all the data
# Returned vector index corresponds to dinstance L``
function corrFuncXY(g::IsingGraphs.IsingGraph, plot = true)
    corr::Vector{Float32} = []
    x = [1:(g.N-2);]
    for L in 1:(g.N-2)
        append!(corr,corrLXY(g,L))
    end

    if plot
        display(pl.plot(x,corr))
    end

    return corr
end

# Tries all pairs, way to expensive for larger grids
function corrFuncPeriodic(g::IsingGraph)
    dict = Dict{Float32,Tuple{Float32,Int32}}()
    for (idx1,state1) in enumerate(g.state)
        for (idx2,state2) in enumerate(g.state)
            (i1,j1) = idxToCoord(idx1,g.N)
            (i2,j2) = idxToCoord(idx2,g.N)
            
            L::Float32 = sqrt((i1-i2)^2+(j1-j2)^2)
            if haskey(dict,L) == false
                dict[L] = (0,0)
            end

            dict[L] = (dict[L][1] + state1*state2, dict[L][2]+1)
        end
    end

    return dict
end

# Parse Correlation Length Data from string in DF
function parseCorrL(corr_dat)
    corrls = []
    for line in corr_dat
        append!(corrls, [eval(Meta.parse(line))] )
    end
    return corrls
end

# Input dataframe and get correlation length data for all temps
function dfToCorrls(df)
    corrls = df[:,2]
    Ts = df[:,1]
    corrls = parseCorrL(corrls)

    return (Ts,corrls)
end

""" Old magnetization stuff """

# Aggregate all Magnetization measurements for a the same temperatures 
function datMAvgT(dat,dpoints = detDPoints(dat))
    temps = dat[:,1]
    Ms = dat[:,3]
    tempit = 1:dpoints:length(temps)
    temps = temps[tempit]
    Ms = [(sum( Ms[(1+(i-1)*dpoints):(i*dpoints)] )/length(tempit)) for i in 1:length(tempit)]

    return (temps,Ms)
end



# Expand measurements in time
function datMExpandTime(dat,dpoints,dpointwait)
    return
end


end


