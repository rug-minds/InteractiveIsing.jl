struct HamiltonianDisplayEntry
    term_index::Int
    term_label::String
    name::Symbol
    source::Symbol
    origin::Symbol
    info::String
    value::Any
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

"""
    HamiltonianParameterPanel(g, layer_idx, display_cell; show_buttons = true)

Selector/display panel for graph state and Hamiltonian visualizations.
Selectable entries are discovered with `hamiltonian_visualizations`. The
selected entry is rendered into `display_cell`, usually the center of the
simulation UI. Set `show_buttons = false` to keep the display behavior while
hiding the left-side selector buttons.
"""
struct HamiltonianParameterPanel <: AbstractPanel
    graph::Any
    layer_idx::Any
    display_cell::Any
    show_buttons::Bool
end

function HamiltonianParameterPanel(graph, layer_idx, display_cell; show_buttons = true)
    return HamiltonianParameterPanel(graph, layer_idx, display_cell, Bool(show_buttons))
end

axis_trait(::Type{HamiltonianParameterPanel}) = HasAxis()
axiskey(::Type{HamiltonianParameterPanel}) = :display_axis
image_trait(::Type{HamiltonianParameterPanel}) = HasImage()

function mount!(panel::HamiltonianParameterPanel, host::WindowHost, cell; kwargs...)
    grid_width = panel.show_buttons ? 240 : 0
    grid = GridLayout(cell, tellheight = false, valign = :top, halign = :left, width = grid_width)
    handle = PanelHandle(panel, host, grid)
    _register_graph_close!(handle, panel.graph)
    handle[:display_grid] = GridLayout(panel.display_cell)
    rowgap!(handle[:display_grid], 8)
    handle[:entries] = _hamiltonian_display_entries(panel.graph)
    handle[:selected] = Observable(1)
    handle[:selector_buttons] = Any[]
    handle[:buttons_hidden] = !panel.show_buttons
    rowgap!(grid, 8)

    if panel.show_buttons
        Label(grid[1, 1], "Fields", fontsize = 13, halign = :left, tellwidth = false)

        for (idx, entry) in enumerate(handle[:entries])
            button_label = _entry_button_label(entry)
            button = Button(
                grid[idx + 1, 1],
                label = button_label,
                fontsize = _entry_button_fontsize(button_label),
                height = 28,
                width = 230,
                tellwidth = false,
                halign = :left,
            )
            push!(handle[:selector_buttons], button)
            register!(handle, on(button.clicks) do _
                handle[:selected][] = idx
                _draw_hamiltonian_entry!(handle)
            end)
        end
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

_term_label(term, idx) = "$(idx) $(_term_type_label(term))"

_term_type_label(term::PolynomialHamiltonian{2}) = "Quadratic"
_term_type_label(term::PolynomialHamiltonian{4}) = "Quartic"
_term_type_label(term::PolynomialHamiltonian{6}) = "Sextic"
_term_type_label(term::PolynomialHamiltonian{8}) = "Octic"
_term_type_label(term::PolynomialHamiltonian{Order}) where {Order} = "PolynomialHamiltonian{$Order}"
_term_type_label(term) = _compact_type_head(typeof(term))

_type_parameter_label(T::Type) = _compact_type_head(T)
_type_parameter_label(value) = string(value)

function _compact_type_head(T::Type)
    text = string(T)
    brace_idx = findfirst(==('{'), text)
    head = isnothing(brace_idx) ? text : text[begin:prevind(text, brace_idx)]
    dot_idx = findlast(==('.'), head)
    return isnothing(dot_idx) ? head : head[nextind(head, dot_idx):end]
end

function _compact_parametric_type(T::DataType)
    head = string(nameof(T))
    params = T.parameters
    isempty(params) && return head
    return string(head, "{", join(_type_parameter_label.(params), ", "), "}")
end

_compact_parametric_type(T::UnionAll) = _compact_type_head(T)

_term_display_entries(term, term_index, term_label, g) =
    [HamiltonianDisplayEntry(term_index, term_label, spec) for spec in hamiltonian_visualizations(term, g)]

function _entry_button_label(entry::HamiltonianDisplayEntry)
    entry.source === :graph && return "Graph state"
    return "$(entry.term_label).$(entry.name)"
end

function _entry_button_fontsize(label; width = 230, max_fontsize = 10, min_fontsize = 7)
    usable_width = 0.88 * width
    estimated_width = max(length(label), 1) * 0.56 * max_fontsize
    estimated_width <= usable_width && return max_fontsize
    return max(min_fontsize, min(max_fontsize, floor(Int, usable_width / (0.56 * length(label)))))
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
    for key in (:display_axis, :display_vector_arrow_magnitude_axis, :display_label, :display_title)
        if haskey(handle, key)
            key === :display_axis && _remember_axis3_state!(handle, :display_axis3_state, handle[key])
            _delete_makie_object!(handle, handle[key])
            delete!(handle.data, key)
        end
    end
    for key in (
        :display_obs,
        :display_entry,
        :display_is_3d,
        :display_vectorized,
        :display_plot,
        :display_use_data_colorrange,
        :display_notify_only,
        :display_vector_arrow_layer,
        :display_vector_arrow_positions,
        :display_vector_arrow_directions,
        :display_vector_arrow_magnitudes,
        :display_vector_arrow_underlay,
        :display_vector_arrow_stem_segments,
        :display_vector_arrow_head_positions,
        :display_vector_arrow_head_rotations,
        :display_vector_arrow_head_sizes,
        :display_vector_arrow_stem_plot,
        :display_vector_arrow_magnitude_axis,
        :display_vector_arrow_magnitude_heatmap_obs,
        :display_vector_arrow_magnitude_heatmap_plot,
        :display_vector_arrow_plot,
    )
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
    elseif val isa LiveLayerDisplayValue
        return _draw_live_layer_display_value!(handle, entry, cell, val, layer)
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
    if _is_vector_spin_2d_layer(layer)
        handle[:display_is_3d] = false
        handle[:display_use_data_colorrange] = false
        handle[:display_notify_only] = true
        return _draw_vector_spin_layer_2d!(
            handle,
            cell,
            layer;
            axis_key = :display_axis,
            prefix = :display_vector_arrow,
            yflip_default = false,
        )
    end

    _draw_layer_array!(
        handle,
        cell,
        _layer_state_values(layer),
        layer;
        colormap = entry.colormap,
        colorrange = _entry_colorrange(entry, _layer_state_values(layer)),
        use_data_colorrange = _uses_data_colorrange(entry),
        hot = true,
    )
    return handle
end

function _draw_graph_state!(handle, entry, cell, layer::AbstractIsingLayer{T,3}) where {T}
    if _is_vector_spin_3d_layer(layer)
        handle[:display_is_3d] = true
        handle[:display_use_data_colorrange] = false
        handle[:display_notify_only] = true
        return _draw_vector_spin_layer_3d!(
            handle,
            cell,
            layer;
            axis_key = :display_axis,
            prefix = :display_vector_arrow,
        )
    end

    ax = handle[:display_axis] = Axis3(cell, tellheight = true)
    _restore_axis3_state!(ax, get(handle.data, :display_axis3_state, nothing))
    xs, ys, zs = _coordinates_3d!(handle, layer)
    obs = handle[:display_obs] = hot_observable!(handle, _cast_layer_state_vector(layer))
    handle[:display_is_3d] = true
    handle[:display_use_data_colorrange] = _uses_data_colorrange(entry)
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
        use_data_colorrange = _uses_data_colorrange(entry),
        hot = true,
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
        use_data_colorrange = _uses_data_colorrange(entry),
    )
end

function _draw_live_layer_display_value!(handle, entry, cell, val::LiveLayerDisplayValue, layer)
    shaped = _layer_values(val, layer)
    _draw_layer_array!(
        handle,
        cell,
        shaped,
        layer;
        colormap = entry.colormap,
        colorrange = _entry_colorrange(entry, shaped),
        use_data_colorrange = _uses_data_colorrange(entry),
        hot = true,
    )
    handle[:display_notify_only] = true
    return handle
end

function _draw_layer_array!(
    handle,
    cell,
    vals,
    layer::AbstractIsingLayer{T,2};
    colormap = :viridis,
    colorrange = nothing,
    use_data_colorrange = false,
    hot = false,
) where {T}
    handle[:display_is_3d] = false
    handle[:display_use_data_colorrange] = use_data_colorrange
    return topology_layer_display!(
        handle,
        cell,
        topology(layer),
        vals,
        layer;
        axis_key = :display_axis,
        obs_key = :display_obs,
        plot_key = :display_plot,
        vectorized_key = :display_vectorized,
        colormap,
        colorrange,
        hot,
        yflip_default = false,
    )
end

function _draw_layer_array!(
    handle,
    cell,
    vals,
    layer::AbstractIsingLayer{T,3};
    colormap = :viridis,
    colorrange = nothing,
    use_data_colorrange = false,
    hot = false,
) where {T}
    vals_size = size(vals)
    length(vals_size) == 3 || throw(ArgumentError("3D layer display needs a 3D array, got size $(vals_size)."))
    ax = handle[:display_axis] = Axis3(cell, tellheight = true)
    _restore_axis3_state!(ax, get(handle.data, :display_axis3_state, nothing))
    xs, ys, zs = vals_size == size(layer) ? _coordinates_3d!(handle, layer) : _coordinates_3d!(handle, vals_size)
    display_vals = vec(vals)
    obs = handle[:display_obs] = hot ? hot_observable!(handle, display_vals) : Observable(display_vals)
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
    haskey(handle, :display_entry) || return nothing

    if !haskey(handle, :display_obs)
        _refresh_vector_spin_arrows!(handle; prefix = :display_vector_arrow)
        return nothing
    end

    if get(handle.data, :display_notify_only, false)
        vals = _entry_layer_values(handle[:display_entry], handle)
        if get(handle.data, :display_use_data_colorrange, false) && haskey(handle, :display_plot) && !isnothing(vals)
            handle[:display_plot].colorrange[] = _entry_colorrange(handle[:display_entry], vals)
        end
        notify(handle[:display_obs])
        _refresh_vector_spin_arrows!(handle; prefix = :display_vector_arrow)
        return nothing
    end

    vals = _entry_layer_values(handle[:display_entry], handle)
    isnothing(vals) && return nothing
    handle[:display_obs][] =
        get(handle.data, :display_vectorized, false) ? vec(vals) :
        handle[:display_is_3d] ? vec(vals) :
        vals
    if get(handle.data, :display_use_data_colorrange, false) && haskey(handle, :display_plot)
        handle[:display_plot].colorrange[] = _entry_colorrange(handle[:display_entry], vals)
    end
    _refresh_vector_spin_arrows!(handle; prefix = :display_vector_arrow)
    return nothing
end

function _entry_layer_values(entry::HamiltonianDisplayEntry, handle)
    return _with_layer(handle.panel.graph, handle.panel.layer_idx) do layer
        _entry_layer_values(entry, handle, layer)
    end
end

function _entry_layer_values(entry::HamiltonianDisplayEntry, handle, layer)
    entry.source === :graph && return _layer_state_values(layer)
    entry.value isa LiveLayerDisplayValue && return _layer_values(entry.value, layer)
    entry.value isa LayerDisplayValue && return _layer_values(entry.value, layer)
    _is_state_sized(entry.value, handle.panel.graph) || return nothing
    return _layer_values(entry.value, layer)
end

_layer_state_values(layer) = copy(_layer_state_view(layer))

function _layer_values(val, layer)
    layer_vals = @view val[graphidxs(layer)]
    return reshape(collect(layer_vals), size(layer))
end

_layer_values(val::LayerDisplayValue, layer) = val.f(layer)
_layer_values(val::LiveLayerDisplayValue, layer) = val.f(layer)

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

function _symmetric_array_colorrange(vals)
    finite_vals = filter(isfinite, vec(vals))
    isempty(finite_vals) && return (-1.0, 1.0)

    hi = maximum(abs, finite_vals)
    hi == 0 && return (-1.0, 1.0)
    return (-hi, hi)
end

_uses_data_colorrange(entry::HamiltonianDisplayEntry) =
    entry.colorrange === :data || entry.colorrange === :symmetric_data

_entry_colorrange(entry::HamiltonianDisplayEntry, vals) =
    entry.colorrange === :data ? _array_colorrange(vals) :
    entry.colorrange === :symmetric_data ? _symmetric_array_colorrange(vals) :
    nothing

function toimage!(cell, panel::HamiltonianParameterPanel, handle::PanelHandle; kwargs...)
    entries = handle[:entries]
    isempty(entries) && throw(ArgumentError("HamiltonianParameterPanel has no display entries."))
    idx = clamp(handle[:selected][], 1, length(entries))
    entry = entries[idx]

    grid = GridLayout(cell)
    Label(grid[1, 1], _display_title(entry), fontsize = 14, tellwidth = false, halign = :center)
    rowsize!(grid, 1, Auto())

    export_handle = PanelHandle(panel, handle.host, grid)
    if haskey(handle, :display_axis)
        _remember_axis3_state!(export_handle, :display_axis3_state, handle[:display_axis])
    else
        export_handle[:display_axis3_state] = get(handle.data, :display_axis3_state, nothing)
    end

    _draw_value!(export_handle, entry, grid[2, 1])
    return grid
end
