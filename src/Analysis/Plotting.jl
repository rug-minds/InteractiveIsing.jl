#TODO: Create a cleaner interface for plotting

"""
Plot and save correlation data from a vec with sampled lengths and vec with correlation data for those lengths
the other arguments are for the file names 
"""
function plotCorr(lVec, corrVec; save = true, dodisplay = true, foldername = "Data", lidx = 0, tidx = 0, point = 0, T = 0)
    corrPlot = pl.plot(lVec,corrVec, xlabel = "Length", label = L"C(L)", ylimits = [-0.5,1])
    
    if dodisplay
        display(corrPlot)
    end

    if save
        Tstring = replace("$T", '.' => ',')
        pl.savefig(corrPlot,"$(foldername)Ising Corrplot L$lidx $tidx T$Tstring d$point")
    end
end

plotCorr(vecs::Tuple; kwargs...) = let (lVec,corrVec) = vecs; plotCorr(lVec,corrVec; kwargs...) end

plotCorr(layer::IsingLayer; save = false, kwargs...) = plotCorr(correlationLength(layer); save, kwargs...)
export plotCorr


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
