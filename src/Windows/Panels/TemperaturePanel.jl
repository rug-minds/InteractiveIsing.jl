struct TemperaturePanel{G} <: AbstractPanel
    graph::G
    slider_max::Float64
end

TemperaturePanel(graph; slider_max = 20.0) = TemperaturePanel(graph, Float64(slider_max))

function mount!(panel::TemperaturePanel, host::WindowHost, cell; kwargs...)
    grid = GridLayout(cell, halign = :center)
    handle = PanelHandle(panel, host, grid)
    g = panel.graph

    Box(grid[1, 1], width = 110, height = 50, visible = false)
    po = register_polled!(handle, PolledObservable(temp(g), _ -> temp(g); setter = x -> temp!(g, x)))
    slider = handle[:slider] = Slider(
        grid[2, 1],
        range = 0.0:0.02:panel.slider_max,
        value = temp(g),
        horizontal = false,
        height = 520,
    )
    slider.value.ignore_equal_values = true
    _set_slider_close!(slider, temp(g))

    register!(handle, on(po) do x
        _set_slider_close!(slider, x)
    end)
    register!(handle, on(slider.value) do x
        if 0.0 <= x <= panel.slider_max
            po[] = x
        end
    end)

    handle[:label_text] = lift(x -> "T: $(round(x, digits = 2))", po)
    handle[:label] = Label(grid[1, 1], handle[:label_text], fontsize = 18)
    return handle
end
