"""
Function that automaticall runs the temperature sweep
"""
function tempSweepInner(g, layeridxs, Temp, M_array; TI = Temp[], TF = 13, TStep = 0.5, dpoints = 6, dpointwait = 5, stepwait = 0, equiwait = 0, saveImg = false, samplingAlgo = Mtl, analysisObs = Observable(true), savelast = true, absvalcorrplot = false)
    analysisObs[] = true

    # Print details
    println("Starting temperature sweep")
    println("Parameters TI: $TI,TF: $TF, TStep: $TStep, dpoints: $dpoints, dpointwait: $dpointwait, stepwait: $stepwait, equiwait: $equiwait, saveImg: $saveImg")
    

    """ Make folder """
    foldername = dataFolderNow("Tempsweep")

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

    #Create data dict
    #TODO: FINISH
    data = TempSweepData(g, layeridxs, length(TRange), dpoints)

    for (tidx, T) in enumerate(TRange)
        
        # Print estimated time left
        tempEstTime(tidx, T, TStep, TF, dpoints, dpointwait, dwait_actual, stepwait, equiwait)

        #=
        Actual analysis 
        =#
        for point in 1:dpoints
            if !(analysisObs[])
                println("Interrupted analysis")
                return
            end

            # Layers to be analysed
            layers = @view g[layeridxs]

            tpointi = time()


            Threads.@threads for stateidx in eachindex(states)
                layer = layers[stateidx]
                lVec, corrVec = correlationLength(layer)
                M = sum(state(layer))
                datapoint = DataPoint(M, lVec, corrVec)
                data[stateidx][tidx][point] = datapoint

                if saveImg && (!savelast || point == dpoints)

                    # Image of ising
                    save(File{format"PNG"}("$(foldername)Ising $tidx d$point T$T.PNG"), gToImg(layer))
                    
                    # Image of correlation plot
                    plotCorr(lVec,corrVec; foldername, tidx, point, T)
                end
            end
            # Print amount of time taken for datapoint
            tpointf = time()
            dtpoint = tpointf-tpointi
            println("Datapoint $point took $dtpoint seconds")

            
            # Calculate average time taken for datapoint
            append!(dwaits,dtpoint)
            dwait_actual = sum(dwait_actual)/length(dwait_actual)

            #Sleep if there is time remaining
            sleep(max(dpointwait-dtpoint,0))
        end
        
    end

    analysisObs[] = false
    
    # Save data
    
    save("$(foldername)data.jld2", "data", data)
    
    println("Temperature sweep done!")
    return
end

function gatherLayerData(layer; save = true, foldername = dataFolderNow("Layer $(layeridx(layer)) data"))
    lidx = layeridx(layer)

    statecopy = copy(state(layer))

    println("Gathering data from layer $lidx")
 
    (lVec,corrVec) = correlationLength(layer)

    M = last(M_array(sim(layer)))

    return M, lvec, corrVec
end

@inline function tempEstTime(tidx, T, TStep, TF, dpoints, dpointwait, dwait_actual, stepwait, equiwait)
    println("Gathering data for temperature $T, $dpoints data points in intervals of $dpointwait seconds")
    waittime = length(T:TStep:TF)*(dpoints*dwait_actual + stepwait)

    Temp[] = Float32(T)

    if tidx == 1 # If first run, wait equiwait seconds for equibrilation
        println("The sweep will take approximately $(equiwait + waittime) seconds")
        if equiwait != 0
            println("Waiting $equiwait seconds for equilibration")
            sleep(equiwait)
        end
    else # Else wait in between temperature steps
        # User feedback of remaining time
        println("Approximately $waittime seconds remaining")
        println("Now waiting $stepwait seconds in between temperatures")
        sleep(stepwait)
    end

end