# All functions that are run from the QML Timer
function timedFunctions(sim)
    layer = currentLayer(sim)
    if runTimedFunctions(sim)[]
        updatesPerFrame(sim)
        # spawnOne(updatesPerFrame, sim.updatingUpf, "", sim)
        spawnOne(magnetization, sim.updatingMag, "", sim)
    end

    checkImgSize(sim, layer, glength(layer), gwidth(layer), qmllength(sim), qmllength(sim))
    spawnOne(updateImg, sim.updatingImg, "UpdateImg", sim)

end

# Timed Functions 
# Updating image of graph
export updateImg
function updateImg(sim)
    sim.img[] = gToImg(currentLayer(sim), colorscheme = colorscheme(sim))
    return
end

# Track number of updates per frame
let avgWindow = 60, updateWindow = zeros(Int64,avgWindow), frames = 0
    global function updatesPerFrame(sim::IsingSim, statelength = length(state(gs(sim)[1])))
        updateWindow = insertShift(updateWindow,sim.params.updates)
        if frames > avgWindow
            upf(sim)[] = Float32(sum(updateWindow)/avgWindow/statelength)
            frames = 0
        end
        frames += 1
        sim.params.updates = 0
    end
end

# Averages M_array over an amount of steps
# Updates magnetization (which is thus the averaged value)
let avg_window = 60, frames = 0
    global function magnetization(sim::IsingSim)
        avg_window = 60 # Averaging window = Sec * FPS, becomes max length of vector
        sim.M_array[] = insertShift(sim.M_array[], sum(state(currentLayer(sim))))
        if frames > avg_window
            M(sim)[] = sum(sim.M_array[])/avg_window 
            frames = 0
        end 
        frames += 1 
    end
end