"""
    StatusPanel(g, layer_idx)

Top-strip panel for the simulation UI. It shows a smoothed process steps/sec
estimate, pause/resume controls for graph processes, and utility buttons for
the selected layer.

The pause label is polled from graph process state, so REPL-side process changes
are reflected in the UI.
"""
struct StatusPanel <: AbstractPanel
    graph::Any
    layer_idx::Any
end

image_trait(::Type{StatusPanel}) = HasImage()

function mount!(panel::StatusPanel, host::WindowHost, cell; kwargs...)
    grid = GridLayout(cell)
    handle = PanelHandle(panel, host, grid)
    g = panel.graph
    _register_graph_close!(handle, g)

    Box(grid[1, 1:3], visible = false)
    colsize!(grid, 1, Relative(1 / 3))
    colsize!(grid, 2, Relative(1 / 3))
    colsize!(grid, 3, Relative(1 / 3))

    ups = handle[:ups] = Observable(0.0)
    upsps = handle[:upsps] = Observable(0.0)
    fps = max(1, round(Int, host.fps))
    last_update = Ref(_total_ticks(g))
    last_time = Ref(time_ns())
    update_deltas = CircularBuffer{Float64}(fps)
    time_deltas = CircularBuffer{Float64}(fps)

    register_frame!(handle) do _
        ticks = _total_ticks(g)
        now = time_ns()
        prev = last_update[]
        delta = ticks >= prev ? ticks - prev : 0
        dt = (now - last_time[]) * 1e-9

        last_update[] = ticks
        last_time[] = now
        dt <= 0 && return nothing

        push!(update_deltas, Float64(delta))
        push!(time_deltas, dt)

        total_time = sum(time_deltas)
        total_time <= 0 && return nothing
        ups[] = sum(update_deltas) / total_time
        upsps[] = ups[] / max(nstates(g), 1)
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

function toimage!(cell, panel::StatusPanel, handle::PanelHandle; kwargs...)
    grid = GridLayout(cell, tellheight = false)
    ups = haskey(handle, :ups) ? round(handle[:ups][], digits = 2) : 0.0
    upsps = haskey(handle, :upsps) ? round(handle[:upsps][], digits = 2) : 0.0
    paused = haskey(handle, :graph_paused) ? handle[:graph_paused][] : _graph_paused(panel.graph)
    Label(grid[1, 1], "Steps per second: $(ups)", fontsize = 12, halign = :left, tellwidth = false)
    Label(grid[2, 1], "Steps/sec/unit: $(upsps)", fontsize = 12, halign = :left, tellwidth = false)
    Label(grid[3, 1], paused ? "Paused" : "Running", fontsize = 12, halign = :left, tellwidth = false)
    if haskey(handle.children, :layer_selector)
        toimage!(grid[4, 1], handle.children[:layer_selector]; kwargs...)
    end
    return grid
end
