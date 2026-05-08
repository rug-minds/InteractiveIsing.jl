"""
    MagnetizationPanel(g, layer_idx)

Bottom panel that polls and displays the selected layer magnetization. It also
contains the defect textbox and selected-layer weight-generator label.
"""
struct MagnetizationPanel <: AbstractPanel
    graph::Any
    layer_idx::Any
end

image_trait(::Type{MagnetizationPanel}) = HasImage()

function mount!(panel::MagnetizationPanel, host::WindowHost, cell; kwargs...)
    grid = GridLayout(cell, tellheight = false, tellwidth = false)
    handle = PanelHandle(panel, host, grid)
    g = panel.graph
    layer_idx = panel.layer_idx

    mid_grid = handle[:mid_grid] = GridLayout(grid[1, 1], tellwidth = false)
    mag = register_polled!(handle, PolledObservable(_magnetization(g, layer_idx), _ -> _magnetization(g, layer_idx)))
    register!(handle, on(layer_idx) do _
        poll!(mag)
    end)
    handle[:m_text] = lift(x -> "Magnetization: $x", mag)
    handle[:m_label] = Label(mid_grid[1, 1], handle[:m_text], fontsize = 18)

    textbox = handle[:defect_textbox] = Textbox(mid_grid[2, 1], placeholder = "% Defect", width = 100, defocus_on_submit = true, reset_on_defocus = true)
    register!(handle, on(textbox.stored_string) do s
        if !isnothing(s)
            parsed = tryparse(Int, s)
            if !isnothing(parsed) && 0 <= parsed <= 100
                _with_layer(g, layer_idx) do layer
                    addRandomDefects!(layer, parsed)
                end
            end
            textbox.stored_string[] = nothing
        end
    end)

    wg_text = lift(i -> _with_layer(layer -> "$(wg(layer))", g, i), layer_idx)
    handle[:wf_label] = Label(mid_grid[0, 1], wg_text, fontsize = 12)
    return handle
end

function toimage!(cell, panel::MagnetizationPanel, handle::PanelHandle; kwargs...)
    grid = GridLayout(cell, tellheight = false)
    mag = _magnetization(panel.graph, panel.layer_idx)
    Label(grid[1, 1], "Magnetization: $(round(mag, digits = 4))", fontsize = 14, halign = :center, tellwidth = false)
    wg_text = _with_layer(layer -> "$(wg(layer))", panel.graph, panel.layer_idx)
    Label(grid[2, 1], wg_text, fontsize = 11, halign = :center, tellwidth = false)
    return grid
end
