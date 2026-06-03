"""
    InteractiveVariablesPanel(g)

Panel that mounts one slider per graph-level interactive variable registered via
`interactivevar!(g, ...)`.
"""
struct InteractiveVariablesPanel <: AbstractPanel
    graph::Any
end

image_trait(::Type{InteractiveVariablesPanel}) = HasImage()

function mount!(panel::InteractiveVariablesPanel, host::WindowHost, cell; kwargs...)
    grid = GridLayout(cell, default_rowgap = 8, tellwidth = false)
    handle = PanelHandle(panel, host, grid)
    g = panel.graph
    _register_graph_close!(handle, g)

    entries = Any[]
    row = 1
    for spec in interactivevars(g)
        spec.varname in (:T, :temp) && continue

        current = _interactive_graph_var_value(g, spec)
        isnothing(current) && continue

        po = register_polled!(
            handle,
            PolledObservable(
                current,
                _ -> _interactive_graph_var_value(g, spec);
                setter = x -> _set_graph_interactive_var!(g, spec, x),
            ),
        )

        label_text = lift(po) do value
            rounded = value isa Integer ? string(value) : string(round(Float64(value); sigdigits = 4))
            return string(spec.label, ": ", rounded)
        end
        label = Label(grid[row, 1], label_text; tellwidth = false, halign = :left, fontsize = 16)
        row += 1

        slider_range = _interactive_slider_range(spec, current)
        slider = Slider(
            grid[row, 1],
            range = slider_range,
            value = current,
            horizontal = true,
            width = 210,
        )
        slider.value.ignore_equal_values = true
        _set_slider_close!(slider, current)
        row += 1

        controls = GridLayout(grid[row, 1], tellwidth = false, default_colgap = 6)
        delta = Observable(_interactive_default_delta(slider_range, current))
        minusbutton = Button(
            controls[1, 1],
            label = "-",
            width = 34,
            height = 28,
            padding = (0, 0, 0, 0),
        )
        deltatext = Textbox(
            controls[1, 2],
            placeholder = string(delta[]),
            width = 72,
            defocus_on_submit = true,
            reset_on_defocus = false,
        )
        plusbutton = Button(
            controls[1, 3],
            label = "+",
            width = 34,
            height = 28,
            padding = (0, 0, 0, 0),
        )
        hold_timer = Ref{Union{Nothing, Timer}}(nothing)

        function stop_hold_timer!()
            timer = hold_timer[]
            isnothing(timer) || close(timer)
            hold_timer[] = nothing
            return nothing
        end

        function apply_step!(direction::Int)
            current_value = convert(typeof(delta[]), _interactive_quantize_value(slider.value[], slider_range))
            step_delta = _interactive_commit_delta!(deltatext, delta, current_value)
            po[] = _interactive_adjusted_value(current_value, step_delta, direction, slider_range)
            return po[]
        end

        function start_hold_timer!(direction::Int)
            stop_hold_timer!()
            hold_timer[] = Timer(0.35; interval = 0.08) do _
                apply_step!(direction)
            end
            return nothing
        end

        minus_mouseevents = addmouseevents!(minusbutton.blockscene, minusbutton.layoutobservables.computedbbox; priority = 2)
        plus_mouseevents = addmouseevents!(plusbutton.blockscene, plusbutton.layoutobservables.computedbbox; priority = 2)

        register!(handle, on(po) do value
            _set_slider_close!(slider, value)
        end)
        register!(handle, on(slider.value) do value
            po[] = value
        end)
        register!(handle, on(deltatext.stored_string) do text
            delta[] = _parse_interactive_delta(text, po[], delta[])
        end)
        register!(handle, onmouseleftdown(minus_mouseevents) do _
            apply_step!(-1)
            start_hold_timer!(-1)
            return Consume(false)
        end)
        register!(handle, onmouseleftdown(plus_mouseevents) do _
            apply_step!(1)
            start_hold_timer!(1)
            return Consume(false)
        end)
        register!(handle, onmouseleftup(minus_mouseevents) do _
            stop_hold_timer!()
            return Consume(false)
        end)
        register!(handle, onmouseleftup(plus_mouseevents) do _
            stop_hold_timer!()
            return Consume(false)
        end)
        register!(handle, onmouseout(minus_mouseevents) do _
            stop_hold_timer!()
            return Consume(false)
        end)
        register!(handle, onmouseout(plus_mouseevents) do _
            stop_hold_timer!()
            return Consume(false)
        end)
        onclose!(handle) do _
            stop_hold_timer!()
        end

        push!(entries, (; spec, label, slider, observable = po, delta, delta_textbox = deltatext, minusbutton, plusbutton, apply_step!))
        row += 1
    end

    handle[:entries] = entries
    return handle
end

function toimage!(cell, panel::InteractiveVariablesPanel, handle::PanelHandle; kwargs...)
    entries = get(handle.data, :entries, Any[])
    isempty(entries) && return Label(cell, "", tellwidth = false)
    labels = map(entry -> string(entry.spec.label), entries)
    return Label(cell, join(labels, "\n"), tellwidth = false, halign = :left, fontsize = 14)
end
