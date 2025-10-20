# Track number of updates per frame
# let avgWindow = 60, updateWindow = AverageCircular(Int, avgWindow), lastTwo = CircularBuffer{UInt}(2) , frames = 1
#     push!(lastTwo, 0)
#     push!(lastTwo, 0)
    
#     global function updatesPerFrame(sim::IsingSim, statelength = length(aliveList(gs(sim)[1])))
#         g = gs(sim)[1]
#         _updates = sum(loopidx.(processes(g)))
#         push!(lastTwo, _updates)
#         push!(updateWindow, Int(lastTwo[2])-Int(lastTwo[1]))

#         if frames > avgWindow
#             sm_avgw = avg(updateWindow)
#             upf(sim)[] = sm_avgw
#             upfps(sim)[] = sm_avgw/statelength
#             frames = 1
#         else
#             frames += 1
#         end
#     end
# end