"""
    TemperaturePanel(g; slider_max = 20.0, slider_step = 0.001)

Panel that displays and edits graph temperature. The value is backed by a
`PolledObservable`, so REPL-side `temp!(g, x)` changes and compatible process
context temperature changes are reflected in the slider. UI-side changes write
back to `temp!(g, x)` and compatible process-context temperature variables.
"""
struct TemperaturePanel <: AbstractPanel
    graph::Any
    slider_max::Float64
    slider_step::Float64
end

image_trait(::Type{TemperaturePanel}) = HasImage()

function TemperaturePanel(graph; slider_max = 20.0, slider_step = 0.001)
    return TemperaturePanel(graph, Float64(slider_max), Float64(slider_step))
end

function mount!(panel::TemperaturePanel, host::WindowHost, cell; kwargs...)
    grid = GridLayout(cell, halign = :center)
    handle = PanelHandle(panel, host, grid)
    g = panel.graph
    _register_graph_close!(handle, g)

    Box(grid[1, 1], width = 150, height = 50, visible = false)
    last_graph_temp = Ref{Any}(temp(g))
    last_context_temp = Ref{Any}(_process_context_temperature(g))
    po = register_polled!(
        handle,
        PolledObservable(
            temp(g),
            _ -> _poll_temperature!(g, last_graph_temp, last_context_temp);
        ),
    )
    slider = handle[:slider] = Slider(
        grid[2, 1],
        range = 0.0:panel.slider_step:panel.slider_max,
        value = temp(g),
        horizontal = false,
        height = 520,
    )
    slider.value.ignore_equal_values = true
    syncing_slider = Ref(false)
    _set_slider_close!(slider, temp(g))

    register!(handle, on(po) do x
        syncing_slider[] = true
        try
            _set_slider_close!(slider, x)
        finally
            syncing_slider[] = false
        end
    end)
    register!(handle, on(slider.value) do x
        syncing_slider[] && return nothing
        if 0.0 <= x <= panel.slider_max
            _set_temperature!(g, x)
            po[] = x
        end
    end)

    handle[:label_text] = lift(x -> _temperature_label(g, x), po)
    handle[:label] = Label(grid[1, 1], handle[:label_text], fontsize = 18)
    return handle
end

function toimage!(cell, panel::TemperaturePanel, handle::PanelHandle; kwargs...)
    value = haskey(handle, :slider) ? handle[:slider].value[] : temp(panel.graph)
    return Label(cell, _temperature_label(panel.graph, value), fontsize = 14, tellwidth = false, halign = :center)
end
