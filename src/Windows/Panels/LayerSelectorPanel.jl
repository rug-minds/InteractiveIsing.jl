"""
    LayerSelectorPanel(g, layer_idx)

Small `< idx/n >` selector for multi-layer graphs. It mutates the observable
`layer_idx`, which lets sibling panels redraw against the selected layer.
"""
struct LayerSelectorPanel <: AbstractPanel
    graph::Any
    layer_idx::Any
end

image_trait(::Type{LayerSelectorPanel}) = HasImage()

function mount!(panel::LayerSelectorPanel, host::WindowHost, cell; kwargs...)
    grid = GridLayout(cell, tellwidth = false)
    handle = PanelHandle(panel, host, grid)
    g = panel.graph
    _register_graph_close!(handle, g)
    layer_idx = panel.layer_idx

    label_text = lift(i -> "$i/$(length(layers(g)))", layer_idx)
    leftbutton = handle[:leftbutton] = Button(grid[1, 1], label = "<", padding = (0, 0, 0, 0), fontsize = 14, width = 40, height = 28)
    handle[:label] = Label(grid[1, 2], label_text, fontsize = 18)
    rightbutton = handle[:rightbutton] = Button(grid[1, 3], label = ">", padding = (0, 0, 0, 0), fontsize = 14, width = 40, height = 28)

    register!(handle, on(leftbutton.clicks) do _
        layer_idx[] > 1 && (layer_idx[] -= 1)
    end)
    register!(handle, on(rightbutton.clicks) do _
        layer_idx[] < length(layers(g)) && (layer_idx[] += 1)
    end)
    return handle
end

function toimage!(cell, panel::LayerSelectorPanel, handle::PanelHandle; kwargs...)
    return Label(
        cell,
        "Layer $(panel.layer_idx[])/$(length(layers(panel.graph)))",
        fontsize = 12,
        tellwidth = false,
        halign = :center,
    )
end
