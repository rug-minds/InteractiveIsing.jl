#=
User functions
=#

"""
Does analysis over multiple temperatures.
Analyzes magnetization and correlation length.
Usage: Goes from current temperature of simulation, to TF, in stepsizes of TStep.
  Every step, a number of datapoints (dpoints) are recorded, with inervals between them of dpointwait
  In between Temperatures there is an optional wait time of stepwait, for equilibration
  There is an initial wait time of equiwait for equilibration
"""
function tempSweep(layer, Temp, M_array; TI = Temp[], TF = 13, TStep = 0.5, dpoints = 6, dpointwait = 5, stepwait = 0, equiwait = 0, saveImg = false, img = Ref([]), corrF = sampleCorrPeriodic, analysisRunning = Observable(true), savelast = true, absvalcorrplot = false)
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
            if !(analysisRunning[])
                println("Interrupted analysis")
                return
            end

            tpointi = time()

            # Get correlation
            (lVec,corrVec) = corrF(layer)

            #=
            Saving dataframes
            =#
            cldf = vcat(cldf, corrToDF((lVec,corrVec) , point, Temp[]) )
            mdf = vcat(mdf, MToDF(last(M_array[]),point, Temp[]) )

            if saveImg && (!savelast || point == dpoints)

                # Image of ising
                save(File{format"PNG"}("$(foldername)Ising $tidx d$point T$T.PNG"), img[])
                
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

    analysisRunning[] = false

    # Try to save the image, but always save the data even if it fails
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