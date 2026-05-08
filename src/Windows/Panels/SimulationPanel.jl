"""
    SimulationPanel(g)

Composite panel for the default Ising simulation interface. It mounts status,
Hamiltonian parameter selection, graph/field display, temperature, and
magnetization controls around graph `g`.
"""
struct SimulationPanel <: AbstractPanel
    graph::Any
end

image_trait(::Type{SimulationPanel}) = HasImage()

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

function toimage!(cell, panel::SimulationPanel, handle::PanelHandle; kwargs...)
    grid = GridLayout(cell, alignmode = Outside(8), default_rowgap = 8)

    if haskey(handle.children, :status)
        toimage!(grid[1, 1], handle.children[:status]; kwargs...)
    end

    midgrid = GridLayout(grid[2, 1], default_colgap = 12)
    if haskey(handle.children, :hamiltonian_parameters)
        toimage!(midgrid[1, 1], handle.children[:hamiltonian_parameters]; kwargs...)
    else
        throw(ArgumentError("SimulationPanel has no mounted visual child to export."))
    end
    if haskey(handle.children, :temperature)
        toimage!(midgrid[1, 2], handle.children[:temperature]; kwargs...)
        colsize!(midgrid, 2, 90)
    end
    colsize!(midgrid, 1, Auto(false))

    if haskey(handle.children, :magnetization)
        toimage!(grid[3, 1], handle.children[:magnetization]; kwargs...)
    end
    return grid
end

"""
    new_interface(g; framerate = 30, polling_rate = 10,
                  size = (1500, 1000),
                  title = "Interactive Ising Simulation") -> WindowHost

Open the Windows-based simulation interface for graph `g`. This is the
implementation behind `interface(g)`.
"""
function new_interface(g; framerate = 30, polling_rate = 10, size = (1500, 1000), title = "Interactive Ising Simulation")
    fig = Figure(; size)
    host = WindowHost(fig; screen = nothing, fps = framerate, polling_rate, open = Observable(true), start_timers = false)
    host[:title] = title
    host[:simulation] = panel!(host, :simulation, SimulationPanel(g), (1, 1))
    _display_host!(host, title)
    return host
end
