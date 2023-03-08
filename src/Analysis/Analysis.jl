# Analysis functions
export dataToDF, tempSweep, MPlot, sampleCorrPeriodic, sampleCorrPeriodicDefects, corrFuncXY, dfMPlot, dataFolderNow, csvToDF, corrPlotDF, corrPlotDFSlices, corrPlotTRange

mutable struct AInt32
    @atomic x::Int32
end

mutable struct AFloat32
    @atomic x::Float32
end

include("User.jl")
include("Plotting.jl")
include("Sampling.jl")
include("Data.jl")

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

# # """ OLD STUFF """

# # Sample random spins for correlation length, but make sure every pair is only sampled once
# # Colissions are quite unlikely for smaller number of sampled pairs, making this redundant and slower
# function sampleCorrPeriodicUnique(g::IsingGraph, Lstep::Float16, lStart::Int32 = 1, lEnd::Int16 = 256, npairs::Integer = 1000 )
#     function sigToIJ(sig, L)
#         return (L*cos(sig),L*sin(sig))
#     end

#     function sampleIdx2(idx1,L,rtheta)
#         ij = idxToCoord(idx1,g.N)
#         dij = Int32.(round.(sigToIJ(rtheta,L)))
#         idx2 = coordToIdx(latmod.((ij.+dij),g.N),g.N)
#         return idx2
#     end

#     theta_i = rand([1:length(rthetas);])

#     avgsum = (sum(g.state)/g.size)^2

#     lVec = [lStart:Lstep:lEnd;]
#     corrVec = Vector{Float32}(undef,length(lVec))

#     pairs =  Set() # To check wether pair is already checked

#     # Iterate over all lengths to be checked
#     for (lidx,L) in enumerate(lVec)
#         pairs_done = 0
 
#         sumprod = 0 #Track the sum of products sig_i*sig_j
#         while pairs_done <= npairs
#             idx1 = rand(g.d.aliveList)
#             rtheta = rthetas[(theta_i -1) % length(rthetas)+1]
#             idx2 = sampleIdx2(idx1,L,rtheta)
            
#             if !((idx1,idx2) in pairs)
#                 sumprod += g.state[idx1]*g.state[idx2]
#                 pairs_done +=1
#                 union!(pairs,(idx1,idx2))
#             else
#                 continue
#             end
#             theta_i += 1 #sample next random angle
#         end
#         # println((sum(g.state)/g.N)^2)
#         # println(avgsum1*avgsum2/(npairs^2))
#         corrVec[lidx] = sumprod/npairs - avgsum
#     end

#     return (lVec,corrVec)

# end

# # Sweep the lattice to find x and y correlation data.
# function corrLXY(g::IsingGraph, L)
#     avgprod = 0
#     prodavg1 = 0
#     prodavg2 = 0
#     Mprod = 0
#     M1 = 0
#     M2 = 0
#     # filter = [ #only do for spin pairs within matrix
#     #     let (i1,j1) = idxToCoord(state,g.N), i2 = i1+L, j2 = j1+L
#     #         i2 <= g.N && j2 <=g.N
#     #     end
#     #     for state in 1:g.size
#     # ]
#     # for state1 in g.state[filter]
#     for stateIdx in g.d.aliveList

#         state1 = g.state[stateIdx]
#         (i1,j1) = idxToCoord(stateIdx,g.N)
#         i2 = i1+L
#         j2 = j1+L

#         # Check if points are added
#         addedi2 = false
#         addedj2 = false

#         if i2 < g.N && !g.d.defectBools[coordToIdx(i2,j1,g.N)]
#             addedi2 = true

#             statey = g.state[coordToIdx(i2,j1,g.N)]
#             prodavg2 += statey   
#             avgprod += state1*statey
#             Mprod += 1
#             M2 +=1
#         end
        
#         if j2 < g.N && !g.d.defectBools[coordToIdx(i1,j2,g.N)]
#             addedj2 = true
#             statex = g.state[coordToIdx(i1,j2,g.N)]
#             prodavg2 += statex
#             avgprod += state1*statex
#             Mprod += 1
#             M2 +=1
#         end

#         if addedi2 || addedj2
#             prodavg1 += state1 
#             M1 += 1
#         end
#     end

#     return avgprod/Mprod-prodavg1*prodavg2/(M1*M2)

# end

# # Calculates the two points correlation function for different lengths and returns a vector with all the data
# # Returned vector index corresponds to dinstance L``
# function corrFuncXY(g::IsingGraph, plot = true)
#     corr::Vector{Float32} = []
#     x = [1:(g.N-2);]
#     for L in 1:(g.N-2)
#         append!(corr,corrLXY(g,L))
#     end

#     if plot
#         display(pl.plot(x,corr))
#     end

#     return corr
# end

# # Tries all pairs, way to expensive for larger grids
# function corrFuncPeriodic(g::IsingGraph)
#     dict = Dict{Float32,Tuple{Float32,Int32}}()
#     for (idx1,state1) in enumerate(g.state)
#         for (idx2,state2) in enumerate(g.state)
#             (i1,j1) = idxToCoord(idx1,g.N)
#             (i2,j2) = idxToCoord(idx2,g.N)
            
#             L::Float32 = sqrt((i1-i2)^2+(j1-j2)^2)
#             if haskey(dict,L) == false
#                 dict[L] = (0,0)
#             end

#             dict[L] = (dict[L][1] + state1*state2, dict[L][2]+1)
#         end
#     end

#     return dict
# end

# # Parse Correlation Length Data from string in DF
# function parseCorrL(corr_dat)
#     corrls = []
#     for line in corr_dat
#         append!(corrls, [eval(Meta.parse(line))] )
#     end
#     return corrls
# end

# # Input dataframe and get correlation length data for all temps
# function dfToCorrls(df)
#     corrls = df[:,2]
#     Ts = df[:,1]
#     corrls = parseCorrL(corrls)

#     return (Ts,corrls)
# end

# # """ Old stuff """

# # Aggregate all Magnetization measurements for a the same temperatures 
# function datMAvgT(dat,dpoints = detDPoints(dat))
#     temps = dat[:,1]
#     Ms = dat[:,3]
#     tempit = 1:dpoints:length(temps)
#     temps = temps[tempit]
#     Ms = [(sum( Ms[(1+(i-1)*dpoints):(i*dpoints)] )/length(tempit)) for i in 1:length(tempit)]

#     return (temps,Ms)
# end



# # Expand measurements in time
# function datMExpandTime(dat,dpoints,dpointwait)
#     return
# end

# # Plots Correlation Length Plot from dataframe
# # Needs the datapoint number and temperature to be plotted
# # Set Absval = true if you want to plot absolute value of correlation length data
# # Is a bit faster than the other method, but doesn't work for variable lengths of lvec and corrvec
# corrPlotDFSlices(filename::String , dpoint, temp, savefolder::String, absval = false) = let cdf = csvToDF(filename); corrPlotDFSlices(cdf, dpoint, temp, savefolder::String, absval) end

# function corrPlotDFSlices(cdf::DataFrame, dpoint, temp, savefolder::String, absval = false)
#     Larray = @view cdf[:,1]
#     corrArray = @view cdf[:,2]
#     dpointArray = @view cdf[:,3]
#     temparray = @view cdf[:,4]
#     l_blocksize = let _ 
#                     len = 1
#                     lastel = Larray[1]
#                     for startidx in 2:length(Larray)
#                         if !(Larray[startidx] < lastel)
#                             len+=1
#                             lastel = Larray[startidx]
#                         else
#                             break
#                         end
#                     end
#                     len
#                 end
#     dpoints = let _
#                 len = 1
#                 lastel = dpointArray[1]
#                 for startidx in (1+l_blocksize):l_blocksize:length(dpointArray)
#                     if !(dpointArray[startidx] < lastel)
#                         len+=1
#                         lastel = dpointArray[startidx]
#                     else
#                         break
#                     end
#                 end
#                 len
#             end

#     tslices = 1:(dpoints*l_blocksize):length(temparray)
    
#     tidx = 0
#     for (startidx,T) in enumerate(@view temparray[tslices])
#         if T == temp
#             tidx = startidx
#         end
#     end
#     if tidx == 0
#         error("T not found")
#         return
#     end
#     startidx = 1+dpoints*l_blocksize*(tidx-1)+l_blocksize*(dpoint-1)
#     endidx = dpoints*l_blocksize*(tidx-1)+l_blocksize*(dpoint-1)+l_blocksize
#     println("T index $tidx")
#     println("Start startidx $startidx")
#     println("End startidx $endidx")
#     x = @view Larray[startidx:endidx]
#     y = @view corrArray[startidx:endidx]

    
#     if absval
#         y = abs.(y)
#     end
    
#     if !absval
#         ylabel = L"C(L)"
#     else
#         ylabel = L"|C(L)|"
#     end

#     cplot = pl.plot(x,y,xlabel = "Length", ylabel=ylabel, label = "T = $temp" )
#     Tstring = replace("$temp", '.' => ',')
#     pl.savefig(cplot,"$(savefolder)Ising Cplot T=$Tstring d$dpoint")
    
# end
