struct SimulationPanel{G} <: AbstractPanel
    graph::G
end

function mount!(panel::SimulationPanel, host::WindowHost, cell; kwargs...)
    grid = GridLayout(cell, alignmode = Outside(24), default_rowgap = 12)
    handle = PanelHandle(panel, host, grid)
    handle[:graph] = panel.graph
    handle[:layer_idx] = Observable(1)

    status = panel!(handle, :status, StatusPanel(panel.graph, handle[:layer_idx]), (1, 1))
    if _has_layer_selector(panel.graph)
        panel!(status, :layer_selector, LayerSelectorPanel(panel.graph, handle[:layer_idx]), status[:selector_slot])
    end

    midgrid = GridLayout(grid[2, 1], default_colgap = 16)
    handle[:midgrid] = midgrid
    panel!(handle, :hamiltonian_parameters, HamiltonianParameterPanel(panel.graph, handle[:layer_idx], midgrid[1, 2]), midgrid[1, 1])
    panel!(handle, :temperature, TemperaturePanel(panel.graph), midgrid[1, 3])
    colsize!(midgrid, 1, 260)
    colsize!(midgrid, 2, Auto(false))
    colsize!(midgrid, 3, 140)

    panel!(handle, :magnetization, MagnetizationPanel(panel.graph, handle[:layer_idx]), (3, 1))
    rowsize!(grid, 3, 140)
    return handle
end

function close!(panel::SimulationPanel, handle::PanelHandle)
    _request_graph_process_close!(panel.graph)
    return nothing
end

function new_interface(g; framerate = 30, polling_rate = 10, size = (1500, 1000), title = "Interactive Ising Simulation")
    host = window(; title, size, fps = framerate, polling_rate)
    host[:simulation] = panel!(host, :simulation, SimulationPanel(g), (1, 1))
    return host
end
