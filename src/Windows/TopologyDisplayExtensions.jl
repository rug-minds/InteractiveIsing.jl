# This page is AI generated.

"""
    Windows.topology_layer_display!(handle, cell, top, vals, layer; kwargs...)

Draw a rectangular two-dimensional topology as a raster image in the Windows
interactive display system.
"""
function Windows.topology_layer_display!(
    handle::Windows.PanelHandle,
    cell,
    top::T,
    vals,
    layer::L;
    axis_key::Symbol,
    obs_key::Symbol,
    plot_key::Symbol,
    vectorized_key::Union{Symbol,Nothing} = nothing,
    colormap = :thermal,
    colorrange = nothing,
    hot::Bool = false,
    yflip_default::Bool = false,
) where {U,T<:AbstractLayerTopology{U,2},L<:AbstractIsingLayer}
    return _raster_topology_layer_display!(
        handle,
        cell,
        vals,
        layer;
        axis_key,
        obs_key,
        plot_key,
        vectorized_key,
        colormap,
        colorrange,
        hot,
        yflip_default,
    )
end

function Windows.topology_layer_display!(
    handle::Windows.PanelHandle,
    cell,
    top::T,
    vals,
    layer::L;
    axis_key::Symbol,
    obs_key::Symbol,
    plot_key::Symbol,
    vectorized_key::Union{Symbol,Nothing} = nothing,
    colormap = :thermal,
    colorrange = nothing,
    hot::Bool = false,
    yflip_default::Bool = false,
) where {U,P,T<:SquareTopology{U,2,P},L<:AbstractIsingLayer}
    return _raster_topology_layer_display!(
        handle,
        cell,
        vals,
        layer;
        axis_key,
        obs_key,
        plot_key,
        vectorized_key,
        colormap,
        colorrange,
        hot,
        yflip_default,
    )
end

function _raster_topology_layer_display!(
    handle::Windows.PanelHandle,
    cell,
    vals,
    layer::L;
    axis_key::Symbol,
    obs_key::Symbol,
    plot_key::Symbol,
    vectorized_key::Union{Symbol,Nothing} = nothing,
    colormap = :thermal,
    colorrange = nothing,
    hot::Bool = false,
    yflip_default::Bool = false,
) where {L<:AbstractIsingLayer}
    vals_size = size(vals)
    length(vals_size) == 2 || throw(ArgumentError("2D topology display needs a matrix, got size $(vals_size)."))
    ax = handle[axis_key] = Axis(cell, xrectzoom = false, yrectzoom = false, aspect = DataAspect(), tellheight = true)
    ax.yreversed = @load_preference("makie_y_flip", default = yflip_default)
    obs = handle[obs_key] = hot ? Windows.hot_observable!(handle, vals) : Observable(vals)
    isnothing(vectorized_key) || (handle[vectorized_key] = false)
    plot = handle[plot_key] = image!(ax, obs, colormap = colormap, fxaa = false, interpolate = false)
    _set_topology_display_colorrange!(plot, obs, layer, colorrange)
    reset_limits!(ax)
    return handle
end

"""
    Windows.topology_layer_display!(handle, cell, top::HexagonalTopology, vals, layer; kwargs...)

Draw a hexagonal topology as one data-space hex marker per axial coordinate.
"""
function Windows.topology_layer_display!(
    handle::Windows.PanelHandle,
    cell,
    top::T,
    vals,
    layer::L;
    axis_key::Symbol,
    obs_key::Symbol,
    plot_key::Symbol,
    vectorized_key::Union{Symbol,Nothing} = nothing,
    colormap = :thermal,
    colorrange = nothing,
    hot::Bool = false,
    yflip_default::Bool = false,
) where {T<:HexagonalTopology,L<:AbstractIsingLayer}
    vals_size = size(vals)
    length(vals_size) == 2 || throw(ArgumentError("Hexagonal topology display needs a matrix, got size $(vals_size)."))
    ax = handle[axis_key] = Axis(cell, xrectzoom = false, yrectzoom = false, aspect = DataAspect(), tellheight = true)
    ax.yreversed = @load_preference("makie_y_flip", default = yflip_default)
    points = _hexagonal_layer_points(top, vals_size)
    display_vals = vec(vals)
    obs = handle[obs_key] = hot ? Windows.hot_observable!(handle, display_vals) : Observable(display_vals)
    isnothing(vectorized_key) || (handle[vectorized_key] = true)

    # Hexagonal topology has skew axes, so render values at topology world
    # coordinates instead of forcing them into a rectangular raster.
    plot = handle[plot_key] = scatter!(
        ax,
        points;
        marker = :hexagon,
        markerspace = :data,
        markersize = _hexagonal_marker_size(top),
        strokewidth = 0,
        color = obs,
        colormap = colormap,
    )
    _set_topology_display_colorrange!(plot, obs, layer, colorrange)
    reset_limits!(ax)
    return handle
end

"""
    _hexagonal_layer_points(top, vals_size)

Return one Makie point per axial coordinate in a hexagonal layer.
"""
function _hexagonal_layer_points(top::T, vals_size::NTuple{2,<:Integer}) where {T<:HexagonalTopology}
    points = Vector{Point2f}(undef, prod(vals_size))
    linear = LinearIndices(vals_size)
    for ci in CartesianIndices(vals_size)
        wc = woorldcoordinate(top, Coordinate(top, ci; check = false))
        points[linear[ci]] = Point2f(Float32(wc[1]), Float32(wc[2]))
    end
    return points
end

"""
    _hexagonal_marker_size(top)

Return the data-space marker diameter used for hexagonal layer displays.
"""
function _hexagonal_marker_size(top::T) where {T<:HexagonalTopology}
    return Float32(0.92 * minimum(lattice_constants(top)))
end

function _set_topology_display_colorrange!(plot, obs, layer::L, colorrange) where {L<:AbstractIsingLayer}
    if isnothing(colorrange)
        Windows._bind_layer_colorrange!(plot, obs, layer)
    else
        plot.colorrange[] = colorrange
    end
    return plot
end
