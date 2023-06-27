#=
User functions
=#

# TODO: Getting a segfault when running this function
"""
Does analysis over multiple temperatures.
Analyzes magnetization and correlation length.
Usage: Goes from current temperature of simulation, to TF, in stepsizes of TStep.
  Every step, a number of datapoints (dpoints) are recorded, with inervals between them of dpointwait
  In between Temperatures there is an optional wait time of stepwait, for equilibration
  There is an initial wait time of equiwait for equilibration
"""
function tempSweep(layer; TI = nothing, TF = 13, TStep = 0.5, dpoints = 6, dpointwait = 5, stepwait = 0, equiwait = 0, saveImg = false, samplingAlgo = Mtl, savelast = true, absvalcorrplot = false)
    println("Starting temperature sweep")
    lsim = sim(layer)
    sTemp = Temp(lsim)
    sM_array = M_array(lsim)
    analysisObs = analysisRunning(lsim)

    if TI === nothing
        TI = sTemp[]
    end

    try
        tempSweepInner(layer, sTemp, sM_array; TI = TI, TF = TF, TStep = TStep, dpoints = dpoints, dpointwait = dpointwait, stepwait = stepwait, equiwait = equiwait, saveImg = saveImg, samplingAlgo = samplingAlgo, analysisObs = analysisObs, savelast = savelast, absvalcorrplot = absvalcorrplot)
    catch
        analysisObs[] = false
    end

end

function tempSweepInner(layer, Temp, M_array; TI = Temp[], TF = 13, TStep = 0.5, dpoints = 6, dpointwait = 5, stepwait = 0, equiwait = 0, saveImg = false, samplingAlgo = Mtl, analysisObs = Observable(true), savelast = true, absvalcorrplot = false)
    analysisObs[] = true

    # Print details
    println("Starting temperature sweep")
    println("Parameters TI: $TI,TF: $TF, TStep: $TStep, dpoints: $dpoints, dpointwait: $dpointwait, stepwait: $stepwait, equiwait: $equiwait, saveImg: $saveImg")
    

    """ Make folder """
    foldername = dataFolderNow("Tempsweep")

    # Constants
    first = true

    # If temperature range goes down, tstep needs to be negative
    if TF < TI
        TStep *= -1
    end

    TRange =  TI:TStep:TF
    
    # DataFrames
    mdf = DataFrame()
    cldf = DataFrame()

    dwaits = []
    dwait_actual = dpointwait
    for (tidx, T) in enumerate(TRange)

        #=
        Printing stuff
        =#
        println("Gathering data for temperature $T, $dpoints data points in intervals of $dpointwait seconds")
        waittime = length(T:TStep:TF)*(dpoints*dwait_actual + stepwait)

        Temp[] = Float32(T)

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

        

        #=
        Actual analysis 
        =#
        for point in 1:dpoints
            if !(analysisObs[])
                println("Interrupted analysis")
                return
            end

            tpointi = time()

            # Get correlation
            (lVec,corrVec) = correlationLength(layer, samplingAlgo)

            #=
            Saving dataframes
            =#
            cldf = vcat(cldf, corrToDF((lVec,corrVec) , point, Temp[]) )
            mdf = vcat(mdf, MToDF(last(M_array[]),point, Temp[]) )

            if saveImg && (!savelast || point == dpoints)

                # Image of ising
                save(File{format"PNG"}("$(foldername)Ising $tidx d$point T$T.PNG"), gToImg(layer))
                
                # Image of correlation plot
                plotCorr(lVec,corrVec; foldername, tidx, point, T)
            end

            tpointf = time()
            dtpoint = tpointf-tpointi
            append!(dwaits,dtpoint)
            dwait_actual = sum(dwait_actual)/length(dwait_actual)
            println("Datapoint $point took $dtpoint seconds")

            sleep(max(dpointwait-dtpoint,0))
        end
        
    end

    analysisObs[] = false
    # Try to save the image, but always save the data even if it fails
    try
        if saveImg
            dfMPlot(mdf, foldername)
        end

    catch(error)
        throw(error)
    finally
        CSV.write("$(foldername)Ising 0 CorrData.csv", cldf)
        CSV.write("$(foldername)Ising 0 MData.csv", mdf)
    end

    println("Temperature sweep done!")
    return
end