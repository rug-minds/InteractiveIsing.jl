"""
    TemperaturePanel(g; slider_max = 20.0)

Panel that displays and edits graph temperature. The value is backed by a
`PolledObservable`, so REPL-side `temp!(g, x)` changes and compatible process
context temperature changes are reflected in the slider. UI-side changes write
back to `temp!(g, x)` and compatible process-context temperature variables.
"""
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
    last_graph_temp = Ref{Any}(temp(g))
    last_context_temp = Ref{Any}(_process_context_temperature(g))
    po = register_polled!(
        handle,
        PolledObservable(
            temp(g),
            _ -> _poll_temperature!(g, last_graph_temp, last_context_temp);
            setter = x -> _set_temperature!(g, x),
        ),
    )
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
