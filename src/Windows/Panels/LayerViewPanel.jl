"""
    LayerViewPanel(g, layer_idx)

Display panel for the selected graph layer. Two-dimensional layers are shown
with `image!`; three-dimensional layers are shown with `meshscatter!`. The 3D
camera orientation is preserved across redraws.
"""
struct LayerViewPanel <: AbstractPanel
    graph::Any
    layer_idx::Any
end

axis_trait(::Type{LayerViewPanel}) = HasAxis()
image_trait(::Type{LayerViewPanel}) = HasImage()

function mount!(panel::LayerViewPanel, host::WindowHost, cell; kwargs...)
    grid = GridLayout(cell)
    handle = PanelHandle(panel, host, grid)
    _register_graph_close!(handle, panel.graph)
    handle[:grid] = grid

    _redraw_layer!(handle)
    register!(handle, on(panel.layer_idx) do _
        _redraw_layer!(handle)
    end)
    register_frame!(handle) do _
        if haskey(handle, :img_obs)
            notify(handle[:img_obs])
        end
        return nothing
    end
    return handle
end

function _redraw_layer!(handle::PanelHandle)
    if haskey(handle, :axis)
        _remember_axis3_state!(handle, :axis3_state, handle[:axis])
        _delete_makie_object!(handle, handle[:axis])
    end

    panel = handle.panel::LayerViewPanel
    grid = handle[:grid]

    _with_layer(panel.graph, panel.layer_idx) do layer
        _draw_layer_view!(handle, grid, layer)
    end
    return handle
end

function _draw_layer_view!(handle, grid, layer::AbstractIsingLayer{T,2}) where {T}
    return topology_layer_display!(
        handle,
        grid[1, 1],
        topology(layer),
        _layer_state_view(layer),
        layer;
        axis_key = :axis,
        obs_key = :img_obs,
        plot_key = :plot,
        colormap = :thermal,
        hot = true,
        yflip_default = true,
    )
end

function _draw_layer_view!(handle, grid, layer::AbstractIsingLayer{T,3}) where {T}
    ax = handle[:axis] = Axis3(grid[1, 1], tellheight = true)
    _restore_axis3_state!(ax, get(handle.data, :axis3_state, nothing))
    xs, ys, zs = _coordinates_3d!(handle, layer)
    vals = _layer_state_vector_view(layer)
    obs = handle[:img_obs] = hot_observable!(handle, vals)
    plot = handle[:plot] = meshscatter!(ax, xs, ys, zs, markersize = 0.3, color = obs, colormap = :thermal)
    _bind_layer_colorrange!(plot, obs, layer)
    return handle
end

function toimage!(cell, panel::LayerViewPanel, handle::PanelHandle; kwargs...)
    return _with_layer(panel.graph, panel.layer_idx) do layer
        _layer_view_toimage!(cell, layer, handle)
    end
end

function _layer_view_toimage!(cell, layer::AbstractIsingLayer{T,2}, handle) where {T}
    ax = Axis(cell, xrectzoom = false, yrectzoom = false, aspect = DataAspect())
    ax.yreversed = @load_preference("makie_y_flip", default = true)
    vals = _layer_state_values(layer)
    plot = image!(ax, vals, colormap = :thermal, fxaa = false, interpolate = false)
    _bind_layer_colorrange!(plot, Observable(vals), layer)
    reset_limits!(ax)
    return ax
end

function _layer_view_toimage!(cell, layer::AbstractIsingLayer{T,3}, handle) where {T}
    ax = Axis3(cell)
    if haskey(handle, :axis)
        _restore_axis3_state!(ax, _axis3_state(handle[:axis]))
    else
        _restore_axis3_state!(ax, get(handle.data, :axis3_state, nothing))
    end
    xs, ys, zs = _coordinates_3d!(handle, layer)
    vals = _cast_layer_state_vector(layer)
    plot = meshscatter!(ax, xs, ys, zs, markersize = 0.3, color = vals, colormap = :thermal)
    _bind_layer_colorrange!(plot, Observable(vals), layer)
    return ax
end
