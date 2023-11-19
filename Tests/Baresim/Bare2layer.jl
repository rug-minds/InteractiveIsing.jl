using InteractiveIsing, JLD2

params = load("examples/baresim/2layer.jld2")
bg = BareGraph(Float32, [(500,500),(250,250)], params["adj"], params["data"].bfield)
timer, task = simulateBare(bg, 1)
function stopbg(bg, task, timer)
    stop(bg)
    try close(timer); wait(task) catch end
    start(bg)
end
