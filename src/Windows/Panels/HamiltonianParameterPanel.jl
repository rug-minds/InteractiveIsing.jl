struct HamiltonianDisplayEntry{V}
    term_index::Int
    term_label::String
    name::Symbol
    source::Symbol
    origin::Symbol
    info::String
    value::V
    colormap::Symbol
    colorrange::Symbol
end

function HamiltonianDisplayEntry(term_index, term_label, spec::HamiltonianDisplaySpec)
    return HamiltonianDisplayEntry(
        term_index,
        term_label,
        spec.name,
        spec.source,
        spec.origin,
        spec.info,
        spec.value,
        spec.colormap,
        spec.colorrange,
    )
end

struct HamiltonianParameterPanel{G,O,C} <: AbstractPanel
    graph::G
    layer_idx::O
    display_cell::C
end

function mount!(panel::HamiltonianParameterPanel, host::WindowHost, cell; kwargs...)
    grid = GridLayout(cell, tellheight = false, valign = :top, halign = :left, width = 240)
    handle = PanelHandle(panel, host, grid)
    handle[:display_grid] = GridLayout(panel.display_cell)
    rowgap!(handle[:display_grid], 8)
    handle[:entries] = _hamiltonian_display_entries(panel.graph)
    handle[:selected] = Observable(1)
    rowgap!(grid, 8)

    Label(grid[1, 1], "Fields", fontsize = 13, halign = :left, tellwidth = false)

    for (idx, entry) in enumerate(handle[:entries])
        button = Button(
            grid[idx + 1, 1],
            label = _entry_button_label(entry),
            fontsize = 10,
            height = 28,
            width = 230,
            tellwidth = false,
            halign = :left,
        )
        register!(handle, on(button.clicks) do _
            handle[:selected][] = idx
            _draw_hamiltonian_entry!(handle)
        end)
    end

    register!(handle, on(panel.layer_idx) do _
        _draw_hamiltonian_entry!(handle)
    end)
    register_frame!(handle) do _
        _refresh_hamiltonian_display!(handle)
    end

    _draw_hamiltonian_entry!(handle)
    return handle
end

function _hamiltonian_display_entries(g)
    entries = HamiltonianDisplayEntry[
        HamiltonianDisplayEntry(
            0,
            "Graph",
            HamiltonianDisplaySpec(
                :state,
                g;
                source = :graph,
                origin = :owned,
                info = "Current graph state",
                colormap = :thermal,
                colorrange = :layer,
            ),
        ),
    ]

    for (term_index, term) in enumerate(_hamiltonian_terms(hamiltonian(g)))
        term_label = _term_label(term, term_index)
        append!(entries, _term_display_entries(term, term_index, term_label, g))
    end

    return entries
end

_hamiltonian_terms(ham::HamiltonianTerms) = collect(hamiltonians(ham))
_hamiltonian_terms(ham) = Any[ham]

_term_label(term, idx) = "$(idx) $(nameof(typeof(term)))"

_term_display_entries(term, term_index, term_label, g) =
    [HamiltonianDisplayEntry(term_index, term_label, spec) for spec in hamiltonian_visualizations(term, g)]

function _entry_button_label(entry::HamiltonianDisplayEntry)
    entry.source === :graph && return "Graph state"
    return "$(entry.term_label).$(entry.name)"
end

function _draw_hamiltonian_entry!(handle::PanelHandle)
    _clear_hamiltonian_display!(handle)

    entries = handle[:entries]
    isempty(entries) && return handle

    idx = clamp(handle[:selected][], 1, length(entries))
    entry = entries[idx]
    grid = handle[:display_grid]
    handle[:display_entry] = entry

    handle[:display_title] = Label(
        grid[1, 1],
        _display_title(entry),
        fontsize = 12,
        tellwidth = false,
        halign = :center,
    )
    rowsize!(grid, 1, Auto())

    _draw_value!(handle, entry, grid[2, 1])
    return handle
end

function _clear_hamiltonian_display!(handle)
    for key in (:display_axis, :display_label, :display_title)
        if haskey(handle, key)
            try
                delete!(handle[key])
            catch
            end
            delete!(handle.data, key)
        end
    end
    for key in (:display_obs, :display_entry, :display_is_3d, :display_plot, :display_use_data_colorrange, :display_notify_only)
        delete!(handle.data, key)
    end
    return handle
end

function _draw_value!(handle, entry::HamiltonianDisplayEntry, cell)
    entry.source === :graph && return _draw_graph_state!(handle, entry, cell)

    val = entry.value
    return _with_layer(handle.panel.graph, handle.panel.layer_idx) do layer
        _draw_value!(handle, entry, cell, val, layer)
    end
end

function _draw_value!(handle, entry::HamiltonianDisplayEntry, cell, val, layer)
    if val isa LayerDisplayValue
        return _draw_layer_display_value!(handle, entry, cell, val, layer)
    elseif _is_state_sized(val, handle.panel.graph)
        return _draw_layer_vector!(handle, entry, cell, val, layer)
    else
        handle[:display_label] = Label(cell, _value_summary(val), fontsize = 12, tellwidth = false)
    end
    return handle
end

function _draw_graph_state!(handle, entry, cell)
    return _with_layer(handle.panel.graph, handle.panel.layer_idx) do layer
        _draw_graph_state!(handle, entry, cell, layer)
    end
end

function _draw_graph_state!(handle, entry, cell, layer::AbstractIsingLayer{T,2}) where {T}
    return _draw_layer_array!(
        handle,
        cell,
        _layer_state_values(layer),
        layer;
        colormap = entry.colormap,
        colorrange = _entry_colorrange(entry, _layer_state_values(layer)),
        use_data_colorrange = entry.colorrange === :data,
    )
end

function _draw_graph_state!(handle, entry, cell, layer::AbstractIsingLayer{T,3}) where {T}
    ax = handle[:display_axis] = Axis3(cell, tellheight = true)
    xs, ys, zs = _old_linear_layer_coordinates(size(layer))
    obs = handle[:display_obs] = Observable(_cast_layer_state_vector(layer))
    handle[:display_is_3d] = true
    handle[:display_use_data_colorrange] = entry.colorrange === :data
    handle[:display_notify_only] = true
    plot = handle[:display_plot] = meshscatter!(
        ax,
        xs,
        ys,
        zs;
        markersize = 0.3,
        color = obs,
        colormap = entry.colormap,
    )
    _set_display_colorrange!(plot, obs, layer, _entry_colorrange(entry, state(layer)))
    return handle
end

function _draw_layer_vector!(handle, entry, cell, val, layer)
    shaped = _layer_values(val, layer)
    return _draw_layer_array!(
        handle,
        cell,
        shaped,
        layer;
        colormap = entry.colormap,
        colorrange = _entry_colorrange(entry, shaped),
        use_data_colorrange = entry.colorrange === :data,
    )
end

function _draw_layer_display_value!(handle, entry, cell, val::LayerDisplayValue, layer)
    shaped = _layer_values(val, layer)
    return _draw_layer_array!(
        handle,
        cell,
        shaped,
        layer;
        colormap = entry.colormap,
        colorrange = _entry_colorrange(entry, shaped),
        use_data_colorrange = entry.colorrange === :data,
    )
end

function _draw_layer_array!(
    handle,
    cell,
    vals,
    layer::AbstractIsingLayer{T,2};
    colormap = :viridis,
    colorrange = nothing,
    use_data_colorrange = false,
) where {T}
    vals_size = size(vals)
    length(vals_size) == 2 || throw(ArgumentError("2D layer display needs a matrix, got size $(vals_size)."))
    ax = handle[:display_axis] = Axis(cell, aspect = DataAspect(), tellheight = true)
    ax.yreversed = @load_preference("makie_y_flip", default = false)
    obs = handle[:display_obs] = Observable(vals)
    handle[:display_is_3d] = false
    handle[:display_use_data_colorrange] = use_data_colorrange
    plot = handle[:display_plot] = image!(ax, obs, colormap = colormap, fxaa = false, interpolate = false)
    _set_display_colorrange!(plot, obs, layer, colorrange)
    reset_limits!(ax)
    return handle
end

function _draw_layer_array!(
    handle,
    cell,
    vals,
    layer::AbstractIsingLayer{T,3};
    colormap = :viridis,
    colorrange = nothing,
    use_data_colorrange = false,
) where {T}
    vals_size = size(vals)
    length(vals_size) == 3 || throw(ArgumentError("3D layer display needs a 3D array, got size $(vals_size)."))
    ax = handle[:display_axis] = Axis3(cell, tellheight = true)
    xs, ys, zs = _old_linear_layer_coordinates(vals_size)
    obs = handle[:display_obs] = Observable(vec(vals))
    handle[:display_is_3d] = true
    handle[:display_use_data_colorrange] = use_data_colorrange
    plot = handle[:display_plot] = meshscatter!(ax, xs, ys, zs, markersize = 0.3, color = obs, colormap = colormap)
    _set_display_colorrange!(plot, obs, layer, colorrange)
    return handle
end

function _value_summary(val)
    if val isa AbstractArray
        return "$(summary(val))\nsize = $(size(val))"
    else
        return summary(val)
    end
end

function _display_title(entry::HamiltonianDisplayEntry)
    entry.source === :graph && return "Graph state"
    return "$(entry.term_label).$(entry.name)"
end

function _refresh_hamiltonian_display!(handle)
    haskey(handle, :display_obs) || return nothing
    haskey(handle, :display_entry) || return nothing

    if get(handle.data, :display_notify_only, false)
        notify(handle[:display_obs])
        return nothing
    end

    vals = _entry_layer_values(handle[:display_entry], handle)
    isnothing(vals) && return nothing
    handle[:display_obs][] = handle[:display_is_3d] ? vec(vals) : vals
    if get(handle.data, :display_use_data_colorrange, false) && haskey(handle, :display_plot)
        handle[:display_plot].colorrange[] = _array_colorrange(vals)
    end
    return nothing
end

function _entry_layer_values(entry::HamiltonianDisplayEntry, handle)
    return _with_layer(handle.panel.graph, handle.panel.layer_idx) do layer
        _entry_layer_values(entry, handle, layer)
    end
end

function _entry_layer_values(entry::HamiltonianDisplayEntry, handle, layer)
    entry.source === :graph && return _layer_state_values(layer)
    entry.value isa LayerDisplayValue && return _layer_values(entry.value, layer)
    _is_state_sized(entry.value, handle.panel.graph) || return nothing
    return _layer_values(entry.value, layer)
end

_layer_state_values(layer) = copy(state(layer))

function _layer_values(val, layer)
    layer_vals = @view val[graphidxs(layer)]
    return reshape(collect(layer_vals), size(layer))
end

_layer_values(val::LayerDisplayValue, layer) = val.f(layer)

function _set_display_colorrange!(plot, obs, layer, colorrange)
    if isnothing(colorrange)
        _bind_layer_colorrange!(plot, obs, layer)
    else
        plot.colorrange[] = colorrange
    end
    return plot
end

function _array_colorrange(vals)
    finite_vals = filter(isfinite, vec(vals))
    isempty(finite_vals) && return (-1.0, 1.0)

    lo, hi = extrema(finite_vals)
    if lo == hi
        δ = max(abs(lo), one(float(lo)))
        return (lo - δ, hi + δ)
    end
    return (lo, hi)
end

_entry_colorrange(entry::HamiltonianDisplayEntry, vals) =
    entry.colorrange === :data ? _array_colorrange(vals) : nothing
