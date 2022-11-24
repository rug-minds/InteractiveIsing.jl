
# Timed Functions 
# Updating image of graph
export updateImg
function updateImg(sim)
    sim.img[] = gToImg(sim.layers[activeLayer(sim)[]])
    return
end

# Track number of updates per frame
let avgWindow = 60, updateWindow = zeros(Int64,avgWindow), frames = 0
    global function updatesPerFrame(sim::IsingSim)
        updateWindow = insertShift(updateWindow,sim.updates)
        if frames > avgWindow
            upf(sim)[] = round(sum(updateWindow)/avgWindow)
            frames = 0
        end
        frames += 1
        sim.updates = 0
    end
end

# Averages M_array over an amount of steps
# Updates magnetization (which is thus the averaged value)
let avg_window = 60, frames = 0
    global function magnetization(sim::IsingSim)
        avg_window = 60 # Averaging window = Sec * FPS, becomes max length of vector
        sim.M_array[] = insertShift(sim.M_array[], sum(sim.layers[activeLayer(sim)[]].state))
        if frames > avg_window
            M(sim)[] = sum(sim.M_array[])/avg_window 
            frames = 0
        end 
        frames += 1 
    end
end