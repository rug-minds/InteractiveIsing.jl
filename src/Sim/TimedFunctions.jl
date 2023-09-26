# All functions that are run from the QML Timer
function timedFunctions(sim)
    # c_layer = currentLayer(sim)
    # if runTimedFunctions(sim)[]
        updatesPerFrame(sim)
        # spawnOne(updatesPerFrame, sim.updatingUpf, "", sim)
        spawnOne(magnetization, sim.updatingMag, "", sim)
    # end

    # checkImgSize(sim, c_layer, glength(c_layer), gwidth(c_layer), qmllength(sim), qmllength(sim))
    # spawnOne(updateImg, sim.updatingImg, "UpdateImg", sim)

end
export timedFunctions

# Timed Functions 
# Updating image of graph
export updateImg
function updateImg(sim)
    sim.img[] = gToImg(currentLayer(sim), colorscheme = colorscheme(sim))
    return
end

# Track number of updates per frame
let avgWindow = 60, updateWindow = CircularBuffer{Int64}(avgWindow) , frames = 0
    global function updatesPerFrame(sim::IsingSim, statelength = length(aliveList(gs(sim)[1])))
        _updates = sum(p_updates.(processes(sim)))
        reset!.(processes(sim))
        push!(updateWindow,_updates)

        if frames > avgWindow
            sm_avgw = sum(updateWindow)/avgWindow
            upf(sim)[] = Float32(sm_avgw)
            upfps(sim)[] = Float32(sm_avgw/statelength)
            frames = 0
        end
        frames += 1
    end
end
export updatesPerFrame

# Averages M_array over an amount of steps
# Updates magnetization (which is thus the averaged value)
let avg_window = 60, frames = 0
    global function magnetization(sim::IsingSim)
        avg_window = 60 # Averaging window = Sec * FPS, becomes max length of vector
        push!(sim.M_array[], sum(state(currentLayer(sim))))
        if frames > avg_window
            M(sim)[] = sum(sim.M_array[])/avg_window 
            frames = 0
        end 
        frames += 1 
    end
end