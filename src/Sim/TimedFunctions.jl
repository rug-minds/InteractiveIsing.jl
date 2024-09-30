# All functions that are run from the QML Timer
# function timedFunctions(sim)
#     updatesPerFrame(sim)
#     @async magnetization(sim)
#     # spawnOne(magnetization, sim.updatingMag, "", sim)
# end
# function timedFunctions(sim, layer)
#     @async updatesPerFrame(sim)
#     magnetization(sim, layer)
#     # spawnOne(magnetization, sim.updatingMag, "", sim)
# end
# export timedFunctions

# Timed Functions 
# Updating image of graph
export updateImg
function updateImg(sim)
    sim.img[] = gToImg(currentLayer(sim), colorscheme = colorscheme(sim))
    return
end

# Track number of updates per frame
let avgWindow = 60, updateWindow = AverageCircular(Int, avgWindow), lastTwo = CircularBuffer{UInt}(2) , frames = 1
    push!(lastTwo, 0)
    push!(lastTwo, 0)
    global function updatesPerFrame(sim::IsingSim, statelength = length(aliveList(gs(sim)[1])))
        g = gs(sim)[1]
        _updates = sum(loopidx.(processes(g)))
        push!(lastTwo, _updates)
        push!(updateWindow, Int(lastTwo[2])-Int(lastTwo[1]))

        if frames > avgWindow
            sm_avgw = avg(updateWindow)
            upf(sim)[] = Float32(sm_avgw)
            upfps(sim)[] = Float32(sm_avgw/statelength)
            frames = 1
        else
            frames += 1
        end
    end
end
# let avgWindow = 60, updateWindow = AverageCircular(Int64,avgWindow) , frames = 1
#     global function updatesPerFrame(sim::IsingSim, statelength = length(aliveList(gs(sim)[1])))
#         _updates = sum(loopidx.(processes(sim)))
#         loopidx.(processes(sim), 0)
#         push!(updateWindow,_updates)
#         if frames > avgWindow
#             sm_avgw = avg(updateWindow)
#             upf(sim)[] = Float32(sm_avgw)
#             upfps(sim)[] = Float32(sm_avgw/statelength)
#             frames = 1
#         else
#             frames += 1
#         end
#     end
# end
export updatesPerFrame

# Averages M_array over an amount of steps
# Updates magnetization (which is thus the averaged value)
let avg_window = 60, frames = 0, M_array = AverageCircular(Float32, avg_window)
    global function magnetization(sim::IsingSim)
        avg_window = 60 # Averaging window = Sec * FPS, becomes max length of vector
        push!(M_array, sum(state(currentLayer(sim))))
        if frames > avg_window
            M(sim)[] = avg(M_array)
            frames = 0
        end 
        frames += 1 
    end
end

let avg_window = 60, frames = 0
    global function magnetization(sim, layer::IsingLayer)
        avg_window = 60 # Averaging window = Sec * FPS, becomes max length of vector
        push!(sim.M_array[], sum(state(layer)))
        if frames > avg_window
            M(sim)[] = sum(sim.M_array[])/avg_window 
            frames = 0
        end 
        frames += 1 
    end
end