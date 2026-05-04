struct StatusPanel{G, O} <: AbstractPanel
    graph::G
    layer_idx::O
end

function mount!(panel::StatusPanel, host::WindowHost, cell; kwargs...)
    grid = GridLayout(cell)
    handle = PanelHandle(panel, host, grid)
    g = panel.graph

    Box(grid[1, 1:3], visible = false)
    colsize!(grid, 1, Relative(1 / 3))
    colsize!(grid, 2, Relative(1 / 3))
    colsize!(grid, 3, Relative(1 / 3))

    ups = handle[:ups] = Observable(0.0)
    upsps = handle[:upsps] = Observable(0.0)
    fps = max(1, round(Int, host.fps))
    last_two_updates = CircularBuffer{Int}(2)
    push!(last_two_updates, _total_ticks(g))
    update_deltas = AverageCircular(Int, fps)
    times = CircularBuffer{UInt64}(fps)
    push!(times, time_ns())

    register_frame!(handle) do _
        ticks = _total_ticks(g)
        push!(last_two_updates, ticks)
        push!(times, time_ns())

        if length(last_two_updates) == 2 && length(times) >= 2
            prev = last_two_updates[1]
            curr = last_two_updates[2]
            delta = curr >= prev ? curr - prev : 0
            push!(update_deltas, delta)
            dt = times[end] - times[1]
            dt == 0 && return nothing

            smoothed_delta = avg(update_deltas)
            ups[] = smoothed_delta / dt * 1e9
            upsps[] = ups[] / max(nstates(g), 1)
        end
        return nothing
    end

    countergrid = GridLayout(grid[1, 1], tellheight = false, tellwidth = false)
    Label(countergrid[1, 1], "Steps per second", padding = (0, 0, 0, 0), fontsize = 12, halign = :left, valign = :top, tellheight = false, tellwidth = false)
    Label(countergrid[2, 1], lift(x -> "$(round(x, digits = 2))", ups), padding = (0, 0, 0, 0), fontsize = 12, halign = :left, valign = :top, tellheight = false, tellwidth = false)
    Label(countergrid[3, 1], "Steps per second, per unit", padding = (0, 0, 0, 0), fontsize = 12, halign = :left, valign = :top, tellheight = false, tellwidth = false)
    Label(countergrid[4, 1], lift(x -> "$(round(x, digits = 2))", upsps), fontsize = 12, halign = :left, valign = :top, tellheight = false, tellwidth = false)

    mid_grid = handle[:mid_grid] = GridLayout(grid[1, 2], tellwidth = false)
    resetbutton = handle[:resetbutton] = Button(mid_grid[1, 1], label = "Reset Graph", fontsize = 18, height = 30, halign = :center, tellwidth = false)
    register!(handle, on(resetbutton.clicks) do _
        nothing
    end)

    graph_paused = handle[:graph_paused] = register_polled!(handle, PolledObservable(_graph_paused(g), _ -> _graph_paused(g)))
    host.paused[] = graph_paused[]
    register!(handle, on(graph_paused) do paused
        host.paused[] = paused
    end)

    buttontext = lift(x -> x ? "Paused" : "Running", graph_paused)
    pausebutton = handle[:pausebutton] = Button(mid_grid[2, 1], padding = (0, 0, 0, 0), fontsize = 18, width = 100, height = 30, label = buttontext, halign = :center, tellwidth = false)
    register!(handle, on(pausebutton.clicks) do _
        if graph_paused[]
            _resume_graph_processes!(g)
        else
            _pause_graph_processes!(g)
        end
        poll!(graph_paused)
    end)
    handle[:selector_slot] = mid_grid[3, 1]

    tools = GridLayout(grid[1, 3], halign = :right)
    cambutton = handle[:cambutton] = Button(tools[1, 1], label = "Cam", padding = (0, 0, 0, 0), fontsize = 14, width = 48, height = 30, halign = :right, valign = :top, tellwidth = false)
    register!(handle, on(cambutton.clicks) do _
        _with_layer(g, panel.layer_idx) do layer
            saveGImg(layer)
        end
    end)

    corrbutton = handle[:corrbutton] = Button(tools[2, 1], label = "Correlation", padding = (0, 0, 0, 0), fontsize = 14, width = 80, height = 30, halign = :right, valign = :top, tellwidth = false)
    register!(handle, on(corrbutton.clicks) do _
        _with_layer(g, panel.layer_idx) do layer
            plotCorr(layer; save = false)
        end
    end)
    return handle
end
