"""
Function that automaticall runs the temperature sweep
"""
function tempSweepInner(
    g, 
    layeridxs, 
    Temp, 
    M_array; 
    TI = Temp[], 
    TF = 13, 
    TStep = 0.5, 
    dpoints = 6, 
    dpointwait = 5, 
    stepwait = 0, 
    equiwait = 0, 
    saveImg = false, 
    analysisObs = Observable(true), 
    savelast = true, 
    absvalcorrplot = false,
    savedata, 
    layer_analysis = (layer, layerdata, idx) -> (), 
    other_analysis = (layercopies, layerdata) -> ()
    )
    
    analysisObs[] = true

    # Print details
    println("Starting temperature sweep")
    println("Parameters TI: $TI,TF: $TF, TStep: $TStep, dpoints: $dpoints, dpointwait: $dpointwait, stepwait: $stepwait, equiwait: $equiwait, saveImg: $saveImg")
    

    """ Make folder """
    foldername = dataFolderNow("TempSweep")

    # If temperature range goes down, tstep needs to be negative
    if TF < TI
        TStep *= -1
    end

    TRange =  TI:TStep:TF

    dwaits = []
    dwait_actual = dpointwait

    # Initialize data
    data = TempSweepData(g, layeridxs, collect(TRange), dpoints)

    for (tidx, T) in enumerate(TRange)
        Temp[] = Float32(T)

        # Print estimated time left and wait
        wait_esttime(tidx, T, TStep, TF, dpoints, dpointwait, dwait_actual, stepwait, equiwait)

        #=
        Actual analysis 
        =#
        for point in 1:dpoints

            if !(analysisObs[])
                println("Interrupted analysis")
                return
            end

            # Layers to be analysed

            tpointi = time()

            # Make copies of all the layers
            layercopies = [IsingLayerCopy(layer) for layer in (@view g[layeridxs])]

            for stateidx in eachindex(layercopies)
            # for stateidx in eachindex(layers)

                layer = layercopies[stateidx]
                lidx = layeridx(layer)
           
                lVec, corrVec = correlationLength(layer)

                M = sum(state(layer))

                datapoint = DataPoint(M, lVec, corrVec)
                
                data[stateidx][tidx][point] = datapoint

                # Other analysis
                layer_analysis(layer, data, stateidx)
                
                if saveImg && (!savelast || point == dpoints)

                    # Image of ising
                    save(File{format"PNG"}("$(foldername)Ising L$lidx $tidx T$T d$point.PNG"), gToImg(layer))
                    
                    # Image of correlation plot
                    plotCorr(lVec,corrVec; foldername, lidx, tidx, point, T)
                end
            end

            # Other analysis
            other_analysis(layercopies, data)
            
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
    
    savedata && save("$(foldername)data.jld2", data)
    
    println("Temperature sweep done!")
    return data
end

@inline function wait_esttime(tidx, T, TStep, TF, dpoints, dpointwait, dwait_actual, stepwait, equiwait)
    println("Gathering data for temperature $T, $dpoints data points in intervals of $dpointwait seconds")
    waittime = length(T:TStep:TF)*(dpoints*dwait_actual + stepwait)

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