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
function tempSweep(sim, layeridxs = 1:length(gs(sim)[1]), TI = nothing, TF = 13, TStep = 0.5, dpoints = 6, dpointwait = 5, stepwait = 0, equiwait = 0, saveImg = false, samplingAlgo = Mtl, savelast = true, absvalcorrplot = false)
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
        tempSweepInner(g, layeridxs, sTemp, sM_array; TI, TF, TStep, dpoints, dpointwait, stepwait, equiwait, saveImg, samplingAlgo, analysisObs, savelast, absvalcorrplot)
    catch
        analysisObs[] = false
    end

end