# Analysis functions
export dataToDF, tempSweep, MPlot, sampleCorrPeriodic, sampleCorrPeriodicDefects, corrFuncXY, dfMPlot, dataFolderNow, csvToDF, corrPlotDF, corrPlotDFSlices, corrPlotTRange

mutable struct AInt32
    @atomic x::Int32
end

mutable struct AFloat32
    @atomic x::Float32
end


#=
User functions
=#

# Does analysis over multiple temperatures.
# Analyzes magnetization and correlation length.
# Usage: Goes from current temperature of simulation, to TF, in stepsizes of TStep.
#   Every step, a number of datapoints (dpoints) are recorded, with inervals between them of dpointwait
#   In between Temperatures there is an optional wait time of stepwait, for equilibration
#   There is an initial wait time of equiwait for equilibration
function tempSweep(g, TIs, M_array; TI = TIs[], TF = 13, TStep = 0.5, dpoints = 6, dpointwait = 5, stepwait = 0, equiwait = 0, saveImg = false, img = Ref([]), corrF = sampleCorrPeriodic, analysisRunning = Observable(true), savelast = true, absvalcorrplot = false)
    
    """ Make folder """
    foldername = dataFolderNow("Tempsweep")

    println("Starting temperature sweep")
    println("Parameters TI: $TI,TF: $TF, TStep: $TStep, dpoints: $dpoints, dpointwait: $dpointwait, stepwait: $stepwait, equiwait: $equiwait, saveImg: $saveImg")
    
    first = true

    if TF < TI
        TStep *= -1
    end

    TRange =  TI:TStep:TF
    
    # DataFrames
    mdf = DataFrame()
    cldf = DataFrame()
    for (tidx, T) in enumerate(TRange)
        println("Gathering data for temperature $T, $dpoints data points in intervals of $dpointwait seconds")
        waittime = length(T:TStep:TF)*(dpoints*dpointwait + stepwait)

        TIs[] = T

        if first # If first run, wait equiwait seconds for equibrilation
            println("The sweep will take approximately $(equiwait + waittime) seconds")
            if equiwait != 0
                println("Waiting $equiwait seconds for equilibration")
                sleep(equiwait)
            end
            first=false
        else # Else wait in between temperature steps
            # User feedback of remaining time
            println("Approximately $waittime seconds remaining")
            println("Now waiting $stepwait seconds in between temperatures")
            sleep(stepwait)
        end

        

        # Gather Datapoints
        for point in 1:dpoints
            if !(analysisRunning[])
                println("Interrupted analysis")
                return
            end

            tpointi = time()
            (lVec,corrVec) = corrF(g)
            cldf = vcat(cldf, corrToDF((lVec,corrVec) , point, TIs[]) )
            mdf = vcat(mdf, MToDF(last(M_array[]),point, TIs[]) )
            if saveImg && (!savelast || point == dpoints)
                # Image of ising
                save(File{format"PNG"}("$(foldername)Ising $tidx d$point T$T.PNG"), img[])
                
                # Image of correlation plot
                plotCorr(lVec,corrVec, foldername, tidx, point, T)
            end

            tpointf = time()
            dtpoint = tpointf-tpointi

            println("Datapoint $point took $dtpoint seconds")

            sleep(max(dpointwait-dtpoint,0))
        end
        
    end

    analysisRunning[] = false

    try
        if saveImg
            dfMPlot(mdf, foldername)
        end

    catch(error)
        error(error)
    finally
        CSV.write("$(foldername)Ising 0 CorrData.csv", cldf)
        CSV.write("$(foldername)Ising 0 MData.csv", mdf)
    end

    println("Temperature sweep done!")
end

function plotCorr(lVec,corrVec, foldername, tidx, point, T)
    corrPlot = pl.plot(lVec,corrVec, xlabel = "Length", label = L"C(L)")
    Tstring = replace("$T", '.' => ',')
    pl.savefig(corrPlot,"$(foldername)Ising Corrplot $tidx d$point T$Tstring")
end

# Correlation length calculations

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

    # Sample all startidx to be used
    # Slight bit faster to do it this way than to sample it every time in the loop
    idx1s = rand(g.d.aliveList,length(lVec)*npairs)
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
            idx1idx += 1 # Sample next random startidx
        end
        # println((sum(g.state)/g.N)^2)
        # println(avgsum1*avgsum2/(npairs^2))
        corrVec[lidx] = sumprod/npairs - avgsum
    end

    return (lVec,corrVec)
end

# Sample correlation length function when there are defects.
function sampleCorrPeriodicDefects(g::IsingGraph, lend = -floor(-sqrt(2)*g.N/2), binsize = .5, npairs::Integer = Int64(round(lend/binsize * 40000)); sig = 1000, periodic = true)
    function torusDist(i1,j1,i2,j2, N)
        dy = abs(i2-i1)
        dx = abs(j2-j1)

        if dy > .5*g.N
            dy = g.N - dy
        end
        if dx > .5*g.N
            dx = g.N - dx
        end

        return sqrt(dx^2+dy^2)
    end
    
    if length(g.d.aliveList) <= 2
        error("Too little alive spins to do analysis")
        return
    end

    idxs1 = rand(g.d.aliveList,npairs)
    idxs2 = rand(g.d.aliveList,npairs)
    lbins = zeros(length(1:binsize:lend))
    lVec = [1:binsize:lend;]
    corrbins = zeros(length(lVec))
    prodavg = sum(g.state[g.d.aliveList])/length(g.d.aliveList)

    for sample in 1:npairs
        idx1 = idxs1[sample]
        idx2 = idxs2[sample]
        
        while idx1 == idx2
            idx2 = rand(g.d.aliveList)
        end

        (i1,j1) = idxToCoord(idx1,g.N)
        (i2,j2) = idxToCoord(idx2,g.N)
        if periodic
            l = torusDist(i1,j1,i2,j2,g.N )
        else
            l = sqrt((i1-i2)^2+(j1-j2)^2)
        end
        
        # println("1 $idx1, 2 $idx2")
        # println("1 $((i1,j1)), 2 $((i2,j2)) ")
        # println("L $l")

        binidx = Int32(floor((l-1)/binsize)+1)
        
        lbins[binidx] += 1
        corrbins[binidx] += g.state[idx1]*g.state[idx2]

    end

    remaining_idxs = []
    for (startidx,pairs_sampled) in enumerate(lbins)
        if pairs_sampled >= sig
            append!(remaining_idxs, startidx)
        end
    end

    corrVec = (corrbins[remaining_idxs] ./ lbins[remaining_idxs]) .- prodavg
    lVec = lVec[remaining_idxs]

    return (lVec,corrVec)
    
end

# Correlation Length Data
# Save correlation date (lVec,corrVec) to dataframe
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


#PLOTTING

function MToDF(M, dpoint::Integer = Int8(1), T::Real = Float16(1))
    return DataFrame(M = M, D = dpoint, T = T)
end

# Plot a magnetization plot from dataframe
function dfMPlot(mdf::DataFrame, foldername::String, filename::String = "")
    """ DF FORMAT """
    """ M : D : T """

    # All magnetizations
    ms = mdf[:,1]
    # All Temperatures
    ts = mdf[:,3]
    slices = []

    lastt = ts[1]
    lasti = 1
    for startidx in 2:length(ts)
        if !(ts[startidx] == lastt)
            append!(slices,[lasti:(startidx-1)])
            lasti=startidx
        end
        lastt = ts[startidx]
    end
    append!(slices, [lasti:length(ts)])

    avgM = Vector{Float32}(undef,length(slices))
    newTs = Vector{Float32}(undef,length(slices))
    for (startidx,slice) in enumerate(slices)
        avgM[startidx] = sum(ms[slice])/length(slice)
        newTs[startidx] = ts[first(slice)]
    end

    if newTs[1] < newTs[end]
        if avgM[1] < 0
            avgM .*= -1
        end
    else
        if avgM[end] < 0
            avgM .*= -1
        end
    end
        

    mPlot = pl.plot(newTs,avgM, xlabel = "T", ylabel="M", label=false)
    filename = replace(filename, '.' => ',')
    pl.savefig(mPlot,"$(foldername)Ising 0 Mplot $filename")
end

dfMPlot(filename::String, foldername::String, savefilename::String) = let df = csvToDF(filename); dfMPlot(df,foldername,savefilename) end

corrPlotDF(filename::String , dpoint, temp; savefolder::String = "", absval = false) = let cdf = csvToDF(filename); corrPlotDF(cdf, dpoint, temp; savefolder, absval) end

function corrPlotDF(cdf::DataFrame, dpoint, temp; savefolder::String = "", absval = false)
    Larray = @view cdf[:,1]
    corrArray = @view cdf[:,2]
    dpointArray = @view cdf[:,3]
    temparray = @view cdf[:,4]

    startidx = 0
    
    for (tidx,T) in enumerate(temparray)
        if T == temp
            startidx += tidx
            break
        end
        if tidx == length(temparray)
            error("Didn't find temperature")
        end
    end


    for (didx,point) in enumerate(@view dpointArray[startidx:end])
        if point == dpoint
            startidx += didx - 1
            break
        end
        if didx == length(@view dpointArray[startidx:end]) || temparray[startidx] != temp
            error("Didn't find datapoint for temp")
        end
    end

    endidx = startidx
    last_l = Larray[startidx]
    for (lidx,llength) in enumerate(@view Larray[(startidx+1):end])
        if llength < last_l
            endidx += lidx - 1
            break
        end
        if lidx == length(@view Larray[(startidx+1):end])
            endidx += lidx
        end
        last_l = llength
    end

    println("Start startidx $startidx")
    println("End endidx $endidx")

    x = @view Larray[startidx:endidx]
    y = @view corrArray[startidx:endidx]

    if absval
        y = abs.(y)
    end
    
    if !absval
        ylabel = L"C(L)"
    else
        ylabel = L"|C(L)|"
    end

    cplot = pl.plot(x,y,xlabel = "Length", ylabel=ylabel, label = "T = $temp" )
    Tstring = replace("$temp", '.' => ',')
    pl.savefig(cplot,"$(savefolder)Ising Cplot T=$Tstring d$dpoint")

end

function corrPlotTRange(filename::String, dpoint, Ts; savefolder = "Data/Images/Correlation/", absval = false)
    trymakefolder(savefolder)
    
    df = csvToDF(filename)
    for T in Ts
        corrPlotDF(df, dpoint, T; savefolder, absval)
    end
end

# General data

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

# Helper Functions 
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

function trymakefolder(foldername::String)
    if foldername[end] == '/'
        foldername = foldername[1:(end-1)]
    end
    try; mkdir(joinpath(dirname(Base.source_path()),foldername)); catch end
end

# """ OLD STUFF """

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
            idx1 = rand(g.d.aliveList)
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
function corrLXY(g::IsingGraph, L)
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
    for stateIdx in g.d.aliveList

        state1 = g.state[stateIdx]
        (i1,j1) = idxToCoord(stateIdx,g.N)
        i2 = i1+L
        j2 = j1+L

        # Check if points are added
        addedi2 = false
        addedj2 = false

        if i2 < g.N && !g.d.defectBools[coordToIdx(i2,j1,g.N)]
            addedi2 = true

            statey = g.state[coordToIdx(i2,j1,g.N)]
            prodavg2 += statey   
            avgprod += state1*statey
            Mprod += 1
            M2 +=1
        end
        
        if j2 < g.N && !g.d.defectBools[coordToIdx(i1,j2,g.N)]
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
function corrFuncXY(g::IsingGraph, plot = true)
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

# """ Old stuff """

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

# Plots Correlation Length Plot from dataframe
# Needs the datapoint number and temperature to be plotted
# Set Absval = true if you want to plot absolute value of correlation length data
# Is a bit faster than the other method, but doesn't work for variable lengths of lvec and corrvec
corrPlotDFSlices(filename::String , dpoint, temp, savefolder::String, absval = false) = let cdf = csvToDF(filename); corrPlotDFSlices(cdf, dpoint, temp, savefolder::String, absval) end

function corrPlotDFSlices(cdf::DataFrame, dpoint, temp, savefolder::String, absval = false)
    Larray = @view cdf[:,1]
    corrArray = @view cdf[:,2]
    dpointArray = @view cdf[:,3]
    temparray = @view cdf[:,4]
    l_blocksize = let _ 
                    len = 1
                    lastel = Larray[1]
                    for startidx in 2:length(Larray)
                        if !(Larray[startidx] < lastel)
                            len+=1
                            lastel = Larray[startidx]
                        else
                            break
                        end
                    end
                    len
                end
    dpoints = let _
                len = 1
                lastel = dpointArray[1]
                for startidx in (1+l_blocksize):l_blocksize:length(dpointArray)
                    if !(dpointArray[startidx] < lastel)
                        len+=1
                        lastel = dpointArray[startidx]
                    else
                        break
                    end
                end
                len
            end

    tslices = 1:(dpoints*l_blocksize):length(temparray)
    
    tidx = 0
    for (startidx,T) in enumerate(@view temparray[tslices])
        if T == temp
            tidx = startidx
        end
    end
    if tidx == 0
        error("T not found")
        return
    end
    startidx = 1+dpoints*l_blocksize*(tidx-1)+l_blocksize*(dpoint-1)
    endidx = dpoints*l_blocksize*(tidx-1)+l_blocksize*(dpoint-1)+l_blocksize
    println("T index $tidx")
    println("Start startidx $startidx")
    println("End startidx $endidx")
    x = @view Larray[startidx:endidx]
    y = @view corrArray[startidx:endidx]

    
    if absval
        y = abs.(y)
    end
    
    if !absval
        ylabel = L"C(L)"
    else
        ylabel = L"|C(L)|"
    end

    cplot = pl.plot(x,y,xlabel = "Length", ylabel=ylabel, label = "T = $temp" )
    Tstring = replace("$temp", '.' => ',')
    pl.savefig(cplot,"$(savefolder)Ising Cplot T=$Tstring d$dpoint")
    
end
