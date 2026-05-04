struct LayerViewPanel{G, O} <: AbstractPanel
    graph::G
    layer_idx::O
end

function mount!(panel::LayerViewPanel, host::WindowHost, cell; kwargs...)
    grid = GridLayout(cell)
    handle = PanelHandle(panel, host, grid)
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
    ax.yreversed = @load_preference("makie_y_flip", default = false)
    obs = handle[:img_obs] = Observable(state(layer))
    plot = handle[:plot] = image!(ax, obs, colormap = :thermal, fxaa = false, interpolate = false)
    _bind_layer_colorrange!(plot, obs, layer)
    reset_limits!(ax)
    return handle
end

function _draw_layer_view!(handle, grid, layer::AbstractIsingLayer{T,3}) where {T}
    ax = handle[:axis] = Axis3(grid[1, 1], tellheight = true)
    _restore_axis3_state!(ax, get(handle.data, :axis3_state, nothing))
    xs, ys, zs = _coordinates_3d!(handle, size(layer))
    obs = handle[:img_obs] = Observable(_cast_layer_state_vector(layer))
    plot = handle[:plot] = meshscatter!(ax, xs, ys, zs, markersize = 0.3, color = obs, colormap = :thermal)
    _bind_layer_colorrange!(plot, obs, layer)
    return handle
end
