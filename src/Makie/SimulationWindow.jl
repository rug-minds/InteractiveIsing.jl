function simwindow(g)
    w = new_window(;window_type = :Simulation, title = "Simulation", objectptr = g)
    w[:layout] = SimLayout(w.f)
    w[:graph] = g
    println("Graph: ", graph(w))

    w[:layer_idx] = Observable(1)
    w[:polling_rate] = 10
    # w[:u_observable] = PolledObservable()

    baseFig(w)
    singleView(w)

    # push!(w.timers, PTimer((timer) -> poll!.(w[:u_observable]), 0., .5))

    return w
end

export simwindow
current_layer(window::MakieWindow{:Simulation}) = graph(window)[window[:layer_idx][]]

graph(window::MakieWindow{:Simulation}) = window.obj_ptr

function Base.show(io::IO, window::MakieWindow{:Simulation})
    println(io, "Simulation Window for:")
    print(graph(window))
end
