"""
    SimulationPanel(g; hide_left_buttons = false)

Composite panel for the default Ising simulation interface. It mounts status,
Hamiltonian parameter selection, graph/field display, temperature, and
magnetization controls around graph `g`. Set `hide_left_buttons = true` to
collapse the left Hamiltonian selector buttons while keeping the central
graph/field display.
"""
struct SimulationPanel <: AbstractPanel
    graph::Any
    hide_left_buttons::Bool
end

function SimulationPanel(graph; hide_left_buttons = false)
    return SimulationPanel(graph, Bool(hide_left_buttons))
end

image_trait(::Type{SimulationPanel}) = HasImage()

function mount!(panel::SimulationPanel, host::WindowHost, cell; kwargs...)
    grid = GridLayout(cell, alignmode = Outside(24), default_rowgap = 12)
    handle = PanelHandle(panel, host, grid)
    handle[:graph] = panel.graph
    handle[:layer_idx] = Observable(1)
    _register_graph_close!(handle, panel.graph)

    status = panel!(handle, :status, StatusPanel(panel.graph, handle[:layer_idx]), (1, 1))
    if _has_layer_selector(panel.graph)
        panel!(status, :layer_selector, LayerSelectorPanel(panel.graph, handle[:layer_idx]), status[:selector_slot])
    end

    midgrid = GridLayout(grid[2, 1], default_colgap = 16)
    handle[:midgrid] = midgrid
    panel!(
        handle,
        :hamiltonian_parameters,
        HamiltonianParameterPanel(
            panel.graph,
            handle[:layer_idx],
            midgrid[1, 2];
            show_buttons = !panel.hide_left_buttons,
        ),
        midgrid[1, 1],
    )
    rightgrid = GridLayout(midgrid[1, 3], tellwidth = false, default_rowgap = 8)
    handle[:rightgrid] = rightgrid
    panel!(handle, :kinetic_time, KineticTimePanel(panel.graph), rightgrid[1, 1])
    panel!(handle, :temperature, TemperaturePanel(panel.graph), rightgrid[2, 1])
    colsize!(midgrid, 1, panel.hide_left_buttons ? 0 : 260)
    colsize!(midgrid, 2, Auto(false))
    colsize!(midgrid, 3, 140)

    panel!(handle, :magnetization, MagnetizationPanel(panel.graph, handle[:layer_idx]), (3, 1))
    rowsize!(grid, 3, 140)
    return handle
end

function close!(panel::SimulationPanel, handle::PanelHandle)
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
        rightgrid = GridLayout(midgrid[1, 2], default_rowgap = 6)
        if haskey(handle.children, :kinetic_time)
            toimage!(rightgrid[1, 1], handle.children[:kinetic_time]; kwargs...)
        end
        toimage!(rightgrid[2, 1], handle.children[:temperature]; kwargs...)
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
                  title = "Interactive Ising Simulation",
                  hide_left_buttons = false) -> WindowHost

Open the Windows-based simulation interface for graph `g`. This is the
implementation behind `interface(g)`. Set `hide_left_buttons = true` to hide
the Hamiltonian selector buttons while keeping the graph/field display.
"""
function new_interface(
    g;
    framerate = 30,
    polling_rate = 10,
    size = (1500, 1000),
    title = "Interactive Ising Simulation",
    hide_left_buttons = false,
)
    fig = Figure(; size)
    host = WindowHost(fig; screen = nothing, fps = framerate, polling_rate, open = Observable(true), start_timers = false)
    host[:title] = title
    _display_host!(host, title)
    host[:simulation] = panel!(host, :simulation, SimulationPanel(g; hide_left_buttons), (1, 1))
    return host
end
