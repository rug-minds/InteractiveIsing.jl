"""
    KineticTimePanel(g)

Compact readout for kinetic Monte Carlo process time. The panel polls graph
process contexts for a KMC time accumulator and shows the current simulation
time, most recent waiting time, and total event rate when available.
"""
struct KineticTimePanel <: AbstractPanel
    graph::Any
end

image_trait(::Type{KineticTimePanel}) = HasImage()

function mount!(panel::KineticTimePanel, host::WindowHost, cell; kwargs...)
    grid = GridLayout(cell, halign = :center, tellheight = false, tellwidth = false)
    handle = PanelHandle(panel, host, grid)
    g = panel.graph
    _register_graph_close!(handle, g)

    Box(grid[1, 1], width = 120, height = 86, visible = false)
    label_text = handle[:label_text] = register_polled!(handle, PolledObservable(_kinetic_time_label(g), _ -> _kinetic_time_label(g)))
    handle[:label] = Label(
        grid[1, 1],
        Observables.observe(label_text);
        fontsize = 12,
        halign = :center,
        valign = :center,
        tellwidth = false,
        tellheight = false,
    )
    return handle
end

function toimage!(cell, panel::KineticTimePanel, handle::PanelHandle; kwargs...)
    label = haskey(handle, :label_text) ? handle[:label_text][] : _kinetic_time_label(panel.graph)
    return Label(cell, label, fontsize = 12, halign = :center, tellwidth = false)
end
