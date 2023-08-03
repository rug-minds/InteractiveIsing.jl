#=
User functions
=#

"""
Get the correlation length function,
should return two vectors x: The start of every length bin, y: The correlation for that length bin
Standard algorithm is algorithm using Metal
"""
correlationLength(layer) = correlationLength(layer, s_algo)
export correlationLength

# TODO: Getting a segfault when running this function
"""
Does analysis over multiple temperatures.
Analyzes magnetization and correlation length.
Usage: Goes from current temperature of simulation, to TF, in stepsizes of TStep.
  Every step, a number of datapoints (dpoints) are recorded, with inervals between them of dpointwait
  In between Temperatures there is an optional wait time of stepwait, for equilibration
  There is an initial wait time of equiwait for equilibration
"""
function tempSweep(
        sim, 
        layeridxs = 1:length(gs(sim)[1]); 
        TI = nothing, 
        TF = 13, 
        TStep = 0.5, 
        dpoints = 6, 
        dpointwait = 5, 
        stepwait = 0, 
        equiwait = 0, 
        saveImg = false, 
        savelast = true, 
        absvalcorrplot = false,
        savadata = true,
        layer_analysis = (layer, layerdata, idx) -> (), 
        other_analysis = (layercopies, layerdata) -> ()
    )
    
    println("Starting temperature sweep")
    sTemp = Temp(sim)
    sM_array = M_array(sim)
    analysisObs = analysisRunning(sim)
    g = gs(sim)[1]

    if TI === nothing
        TI = sTemp[]
    end


    #Function barrier
    try
        tempSweepInner(g, layeridxs, sTemp, sM_array; TI, TF, TStep, dpoints, dpointwait, stepwait, equiwait, saveImg, analysisObs, savelast, absvalcorrplot, savedata, layer_analysis, other_analysis)
    catch
        rethrow()
        analysisObs[] = false
    end

end
export tempSweep

"""
Save tempsweep data to a file in the julia file format
Standard folder is data folder with subfolder that shows the time of creation
"""
savesweep(tsd::TempSweepData; foldername::String = dataFolderNow("TempSweepData")) = save("$(foldername)data.jld2", tsd)
"""
Returns TempSweepData struct from a path to a JLD2 file
"""
opensweep(filename::String) = TempSweepData(load(filename))
export savesweep, opensweep