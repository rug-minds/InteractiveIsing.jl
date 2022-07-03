# Analysis functions

__precompile__()

module Analysis

using FileIO, Images
import Plots as pl
# include("IsingGraphs.jl")
using ..IsingGraphs

include("SquareAdj.jl")
using .SquareAdj

export magnetization, dataToDF, tempSweep, MPlot, sampleCorrPeriodic, corrFunc

"""
User functions
"""

# Does analysis over multiple temperatures.
# Analyzes magnetization and correlation length.
# Usage: Goes from current temperature of simulation, to TF, in stepsizes of TStep.
#   Every step, a number of datapoints (dpoints) are recorded, with inervals between them of dpointwait
#   In between Temperatures there is an optional wait time of stepwait, for equilibration
#   There is an initial wait time of equiwait for equilibration
function tempSweep(g, TIs, M_array; TI = TIs[], TF = 13, TStep = 0.5, dpoints = 12, dpointwait = 5, stepwait = 0, equiwait = 0, saveImg = false, img = Ref([]))
    println("Starting temperature sweep")
    println("Parameters TIs$TIs, M_array$M_array, TI$TI,TF$TF, TStep$TStep, dpoints$dpoints, dpointwait$dpointwait, stepwait$stepwait, equiwait$equiwait, saveImg$saveImg")
    return
    first = true
    # Data 
    Ts = []
    corrls = []
    Ms = []
    TRange =  TI:TStep:TF
    

    for T in TRange
        println("Gathering data for temperature $T, $dpoints data points in intervals of $dpointwait seconds")
        waittime = length(TRange)*(dpoints*dpointwait + stepwait)
        if first # If first run, wait equiwait seconds for equibrilation
            println("The sweep will take approximately $(equiwait + waittime) seconds")
            if equiwait != 0
                println("Waiting $equiwait seconds for equilibration")
                sleep(equiwait)
            end
            first=false
        else # Else wait in between temperatuer steps
            println("Waiting $stepwait seconds in between temperature steps")
            sleep(stepwait)
        end

        # User feedback of remaining time
        println("Approximately $(length(T:TStep:TF)*dpoints*dpointwait + length(T:TStep:TF)*stepwait) seconds remaining")
        
        TIs[] = T

        # Gather Datapoints
        for point in 1:dpoints
            append!(Ts, T)
            append!(corrls,[corrFunc(IsingGraph(g),false)])
            append!(Ms, last(M_array))

            if saveImg
                save(File{format"PNG"}("Images/TempSweep/Ising T$T d$point.PNG"), img[])
            end

            sleep(dpointwait)
        end
        
    end

    return (Ts,corrls,Ms)
end

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

# function sampleCorrPeriodic(g::IsingGraph,Lstep::Float32)
#     function sigToIJ(sig, L)
#         return (L*cos(sig),L*sin(sig))
#     end

#     # Lvec = [1:Lstep:g.N;]
#     Lvec = [99:Lstep:100;]
#     avgsum = (sum(g.state)/g.size)^2
#     corrVec = Vector{Float32}(undef,length(Lvec))
    
#     # ijprob::Int32 = 1/2
#     for (lidx, L) in enumerate(Lvec)
#         weight = L/g.N
#         sig_samples::Int32 = round(2*L)
#         dijs = map((tup) -> (round(Int16, tup[1]),round(Int16, tup[2])) , sigToIJ.([rand()*2*pi for i in 1:sig_samples],L))
#         idxs = [i for i in 1:g.size if rand([true,false,false, false]) ]
#         ijs = idxToCoord.(idxs,g.N)

#         fac = 0
#         # println(length(ijs))
#         # println(dijs)
#         prod = 0
#         # println(ijs)
#         # println(dijs)
#         for ij in ijs
#             for dij in dijs
#                 fac += 1
#                 prod += g.state[coordToIdx(ij,g.N)]*g.state[coordToIdx(latmod.((ij.+dij),g.N),g.N)]
#             end
#         end
#         println("Fac$fac")
#         println("avgsum$avgsum")
#         println("prodprod")
#         corrVec[lidx] = prod/fac-avgsum
#     end

#     return (Lvec,corrVec)
# end
rthetas = 2*pi .* rand(10^7)

function sampleCorrPeriodic(g::IsingGraph,Lstep::Float32, lStart = 1, lEnd = 512, npairs = 1000 )
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
function corrL(g::IsingGraphs.IsingGraph, L)
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
function corrFunc(g::IsingGraphs.IsingGraph, plot = true)
    corr::Vector{Float32} = []
    x = [1:(g.N-2);]
    for L in 1:(g.N-2)
        append!(corr,corrL(g,L))
    end

    if plot
        display(pl.plot(x,corr))
    end

    return corr
end


# Aggregate all Magnetization measurements for a the same temperatures 
function datMAvgT(dat,dpoints = detDPoints(dat))
    temps = dat[:,1]
    Ms = dat[:,3]
    tempit = 1:dpoints:length(temps)
    temps = temps[tempit]
    Ms = [(sum( Ms[(1+(i-1)*dpoints):(i*dpoints)] )/length(tempit)) for i in 1:length(tempit)]

    return (temps,Ms)
end

# Make a plot of the magnetization from DF
function MPlot(fileName, pDefects)
    df = DataFrame(CSV.File(fileName))
    dat = datMAvgT(df)
    
    if dat[2][1] < 0
        dat = (dat[1],[-x for x in dat[2] ])
    end
    display(pl.plot(dat, title = "$pDefects% defects", xlabel = "Temperature" , ylabel = "Magnetization", legend = false))

end

# Expand measurements in time
function datMExpandTime(dat,dpoints,dpointwait)
    return
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


"""Correlation Length Data"""
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

# Not used currently, used to fit correleation length data to a function f
function fitCorrl(dat,dom_end, f, params...)
    dom = Domain(1.:dom_end)
    data = Measures(Vector{Float64}(dat[1:dom_end]),0.)
    model = Model(:comp1 => FuncWrap(f,params...))
    prepare!(model,dom, :comp1)
    return fit!(model,data)
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
    return DataFrame(Temperature = tsData[1], Correlation_Function = [corrL[1:lMax] for corrL in  tsData[2]], Magnetization = tsData[3] )
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


end
