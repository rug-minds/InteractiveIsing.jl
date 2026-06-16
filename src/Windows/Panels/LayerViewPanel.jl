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
    ax = handle[:axis] = Axis(grid[1, 1], xrectzoom = false, yrectzoom = false, aspect = DataAspect(), tellheight = true)
    ax.yreversed = @load_preference("makie_y_flip", default = true)

    # The graph panel owns axis construction; topology dispatch owns geometry.
    return fill_topology_layer_axis!(
        handle,
        ax,
        topology(layer),
        _layer_state_view(layer),
        layer;
        obs_key = :img_obs,
        plot_key = :plot,
        colormap = :thermal,
        hot = true,
    )
end

function _draw_layer_view!(handle, grid, layer::AbstractIsingLayer{T,3}) where {T}
    ax = handle[:axis] = Axis3(grid[1, 1], tellheight = true)
    _restore_axis3_state!(ax, get(handle.data, :axis3_state, nothing))

    # The graph panel owns the 3D axis and camera; topology fills geometry.
    return fill_topology_layer_axis!(
        handle,
        ax,
        topology(layer),
        _layer_state_view(layer),
        layer;
        obs_key = :img_obs,
        plot_key = :plot,
        colormap = :thermal,
        hot = true,
    )
end

function toimage!(cell, panel::LayerViewPanel, handle::PanelHandle; kwargs...)
    return _with_layer(panel.graph, panel.layer_idx) do layer
        _layer_view_toimage!(cell, layer, handle)
    end
end

function _layer_view_toimage!(cell, layer::AbstractIsingLayer{T,2}, handle) where {T}
    export_handle = PanelHandle(handle.panel, handle.host, cell)
    ax = export_handle[:axis] = Axis(cell, xrectzoom = false, yrectzoom = false, aspect = DataAspect())
    ax.yreversed = @load_preference("makie_y_flip", default = true)
    vals = _layer_state_values(layer)

    # Export uses the same topology geometry path as the live layer panel.
    fill_topology_layer_axis!(
        export_handle,
        ax,
        topology(layer),
        vals,
        layer;
        obs_key = :img_obs,
        plot_key = :plot,
        colormap = :thermal,
    )
    return ax
end

function _layer_view_toimage!(cell, layer::AbstractIsingLayer{T,3}, handle) where {T}
    axis3_state = haskey(handle, :axis) ? _axis3_state(handle[:axis]) : get(handle.data, :axis3_state, nothing)
    export_handle = PanelHandle(handle.panel, handle.host, cell)
    ax = export_handle[:axis] = Axis3(cell)
    _restore_axis3_state!(ax, axis3_state)

    # Export follows the same graph-panel axis ownership as live display.
    fill_topology_layer_axis!(
        export_handle,
        ax,
        topology(layer),
        state(layer),
        layer;
        obs_key = :img_obs,
        plot_key = :plot,
        colormap = :thermal,
        display_vals = _cast_layer_state_vector(layer),
    )
    return export_handle[:axis]
end
