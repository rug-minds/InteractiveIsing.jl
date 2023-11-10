using InteractiveIsing, JLD2
params = load("examples/baresim/MNIST_params.jld2")
bg = BareGraph(Float32, [(28,28),(32,32),(2,5)], params["adj"], params["data"].bfield)
timer, task = simulateBare(bg, 1)
function stopbg(bg, task, timer)
    stop(bg)
    try close(timer); wait(task) catch end
    start(bg)
end