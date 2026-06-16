# This page is AI generated.

"""
    Windows.fill_topology_layer_axis!(handle, ax, top, vals, layer; kwargs...)

Fill an existing two-dimensional Windows axis with a rectangular raster view.
"""
function Windows.fill_topology_layer_axis!(
    handle::Windows.PanelHandle,
    ax,
    top::T,
    vals,
    layer::L;
    obs_key::Symbol,
    plot_key::Symbol,
    vectorized_key::Union{Symbol,Nothing} = nothing,
    colormap = :thermal,
    colorrange = nothing,
    hot::Bool = false,
    display_vals = nothing,
    markersize = 0.3f0,
) where {U,T<:AbstractLayerTopology{U,2},L<:AbstractIsingLayer}
    return _raster_topology_layer_display!(
        handle,
        ax,
        vals,
        layer;
        obs_key,
        plot_key,
        vectorized_key,
        colormap,
        colorrange,
        hot,
        display_vals,
        markersize,
    )
end

function Windows.fill_topology_layer_axis!(
    handle::Windows.PanelHandle,
    ax,
    top::T,
    vals,
    layer::L;
    obs_key::Symbol,
    plot_key::Symbol,
    vectorized_key::Union{Symbol,Nothing} = nothing,
    colormap = :thermal,
    colorrange = nothing,
    hot::Bool = false,
    display_vals = nothing,
    markersize = 0.3f0,
) where {U,P,T<:SquareTopology{U,2,P},L<:AbstractIsingLayer}
    return _raster_topology_layer_display!(
        handle,
        ax,
        vals,
        layer;
        obs_key,
        plot_key,
        vectorized_key,
        colormap,
        colorrange,
        hot,
        display_vals,
        markersize,
    )
end

function _raster_topology_layer_display!(
    handle::Windows.PanelHandle,
    ax,
    vals,
    layer::L;
    obs_key::Symbol,
    plot_key::Symbol,
    vectorized_key::Union{Symbol,Nothing} = nothing,
    colormap = :thermal,
    colorrange = nothing,
    hot::Bool = false,
    display_vals = nothing,
    markersize = 0.3f0,
) where {L<:AbstractIsingLayer}
    vals_size = size(vals)
    length(vals_size) == 2 || throw(ArgumentError("2D topology display needs a matrix, got size $(vals_size)."))
    obs = handle[obs_key] = hot ? Windows.hot_observable!(handle, vals) : Observable(vals)
    isnothing(vectorized_key) || (handle[vectorized_key] = false)
    plot = handle[plot_key] = image!(ax, obs, colormap = colormap, fxaa = false, interpolate = false)
    _set_topology_display_colorrange!(plot, obs, layer, colorrange)
    reset_limits!(ax)
    return handle
end

"""
    Windows.fill_topology_layer_axis!(handle, ax, top::HexagonalTopology, vals, layer; kwargs...)

Fill an existing two-dimensional Windows axis with one data-space hex marker
per axial coordinate.
"""
function Windows.fill_topology_layer_axis!(
    handle::Windows.PanelHandle,
    ax,
    top::T,
    vals,
    layer::L;
    obs_key::Symbol,
    plot_key::Symbol,
    vectorized_key::Union{Symbol,Nothing} = nothing,
    colormap = :thermal,
    colorrange = nothing,
    hot::Bool = false,
    display_vals = nothing,
    markersize = 0.3f0,
) where {T<:HexagonalTopology,L<:AbstractIsingLayer}
    vals_size = size(vals)
    length(vals_size) == 2 || throw(ArgumentError("Hexagonal topology display needs a matrix, got size $(vals_size)."))
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
    Windows.fill_topology_layer_axis!(handle, ax, top::LatticeTopology, vals, layer; kwargs...)

Fill an existing two-dimensional Windows axis with a general lattice topology
at its world-coordinate layout. This is needed for staggered layouts where a
raster would hide the geometry.
"""
function Windows.fill_topology_layer_axis!(
    handle::Windows.PanelHandle,
    ax,
    top::T,
    vals,
    layer::L;
    obs_key::Symbol,
    plot_key::Symbol,
    vectorized_key::Union{Symbol,Nothing} = nothing,
    colormap = :thermal,
    colorrange = nothing,
    hot::Bool = false,
    display_vals = nothing,
    markersize = 0.3f0,
) where {Kind,Layout,U,P,T<:LatticeTopology{Kind,Layout,U,2,P},L<:AbstractIsingLayer}
    vals_size = size(vals)
    length(vals_size) == 2 || throw(ArgumentError("Lattice topology display needs a matrix, got size $(vals_size)."))
    points = _topology_world_layer_points(top, vals_size)
    display_vals = vec(vals)
    obs = handle[obs_key] = hot ? Windows.hot_observable!(handle, display_vals) : Observable(display_vals)
    isnothing(vectorized_key) || (handle[vectorized_key] = true)

    # General lattice displays use world coordinates so direct, oblique, and
    # staggered layouts all render according to the topology geometry.
    plot = handle[plot_key] = scatter!(
        ax,
        points;
        marker = _lattice_marker(Kind),
        markerspace = :data,
        markersize = _lattice_marker_size(top),
        strokewidth = 0,
        color = obs,
        colormap = colormap,
    )
    _set_topology_display_colorrange!(plot, obs, layer, colorrange)
    reset_limits!(ax)
    return handle
end

"""
    Windows.fill_topology_layer_axis!(handle, ax, top::AbstractLayerTopology{<:Any,3}, vals, layer; kwargs...)

Fill an existing three-dimensional Windows axis with topology-specific
coordinates and marker scale.
"""
function Windows.fill_topology_layer_axis!(
    handle::Windows.PanelHandle,
    ax,
    top::T,
    vals,
    layer::L;
    obs_key::Symbol,
    plot_key::Symbol,
    vectorized_key::Union{Symbol,Nothing} = nothing,
    colormap = :thermal,
    colorrange = nothing,
    hot::Bool = false,
    display_vals = nothing,
    markersize = _topology_3d_marker_size(top, size(vals)),
) where {U,T<:AbstractLayerTopology{U,3},L<:AbstractIsingLayer}
    vals_size = size(vals)
    length(vals_size) == 3 || throw(ArgumentError("3D topology display needs a 3D array, got size $(vals_size)."))

    xs, ys, zs = Windows._coordinates_3d!(handle, top, vals_size)
    color_vals = isnothing(display_vals) ? vec(vals) : display_vals
    length(color_vals) == prod(vals_size) ||
        throw(ArgumentError("3D topology display needs $(prod(vals_size)) color values, got $(length(color_vals))."))
    obs = handle[obs_key] = hot ? Windows.hot_observable!(handle, color_vals) : Observable(color_vals)
    isnothing(vectorized_key) || (handle[vectorized_key] = true)

    # Topology owns coordinates and marker scale; the graph panel owns the axis.
    plot = handle[plot_key] = meshscatter!(
        ax,
        xs,
        ys,
        zs;
        markersize,
        color = obs,
        colormap = colormap,
    )
    _set_topology_display_colorrange!(plot, obs, layer, colorrange)
    return handle
end

"""
    Windows.fill_topology_layer_axis!(handle, ax, top::LatticeTopology{Hexagonal,<:Any,<:Any,3}, vals, layer; kwargs...)

Fill an existing three-dimensional Windows axis for a hexagonal lattice
topology.
"""
function Windows.fill_topology_layer_axis!(
    handle::Windows.PanelHandle,
    ax,
    top::T,
    vals,
    layer::L;
    obs_key::Symbol,
    plot_key::Symbol,
    vectorized_key::Union{Symbol,Nothing} = nothing,
    colormap = :thermal,
    colorrange = nothing,
    hot::Bool = false,
    display_vals = nothing,
    markersize = _topology_3d_marker_size(top, size(vals)),
) where {Layout,U,P,T<:LatticeTopology{Hexagonal,Layout,U,3,P},L<:AbstractIsingLayer}
    vals_size = size(vals)
    length(vals_size) == 3 || throw(ArgumentError("3D topology display needs a 3D array, got size $(vals_size)."))

    xs, ys, zs = Windows._coordinates_3d!(handle, top, vals_size)
    color_vals = isnothing(display_vals) ? vec(vals) : display_vals
    length(color_vals) == prod(vals_size) ||
        throw(ArgumentError("3D topology display needs $(prod(vals_size)) color values, got $(length(color_vals))."))
    obs = handle[obs_key] = hot ? Windows.hot_observable!(handle, color_vals) : Observable(color_vals)
    isnothing(vectorized_key) || (handle[vectorized_key] = true)

    # The topology owns the embedding; meshscatter only consumes the resulting
    # display-coordinate positions and the flattened layer values.
    plot = handle[plot_key] = meshscatter!(
        ax,
        xs,
        ys,
        zs;
        markersize,
        color = obs,
        colormap = colormap,
    )
    _set_topology_display_colorrange!(plot, obs, layer, colorrange)
    return handle
end

"""
    Windows._coordinates_3d!(handle, top::SquareTopology{<:Any,3}, vals_size)

Return the legacy integer-grid display coordinates for three-dimensional square
topologies.
"""
function Windows._coordinates_3d!(
    handle,
    top::T,
    vals_size::NTuple{3,<:Integer},
) where {U,P,T<:SquareTopology{U,3,P}}
    isnothing(handle) && return Windows._old_linear_layer_coordinates(vals_size)
    return Windows._coordinates_3d!(handle, vals_size)
end

"""
    Windows._coordinates_3d!(handle, top::LatticeTopology{Hexagonal,<:Any,<:Any,3}, vals_size)

Return topology-world display coordinates for three-dimensional hexagonal
lattice topologies.
"""
function Windows._coordinates_3d!(
    handle,
    top::T,
    vals_size::NTuple{3,<:Integer},
) where {Layout,U,P,T<:LatticeTopology{Hexagonal,Layout,U,3,P}}
    return Windows._world_layer_coordinates_3d(top, vals_size)
end

"""
    _hexagonal_layer_points(top, vals_size)

Return one Makie point per axial coordinate in a hexagonal layer.
"""
function _hexagonal_layer_points(top::T, vals_size::NTuple{2,<:Integer}) where {T<:HexagonalTopology}
    return _topology_world_layer_points(top, vals_size)
end

"""
    _topology_world_layer_points(top, vals_size)

Return one Makie point per coordinate using topology world coordinates.
"""
function _topology_world_layer_points(top::T, vals_size::NTuple{2,<:Integer}) where {T<:AbstractLayerTopology}
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
    return Float32(_topology_marker_fill_factor(T) * minimum(lattice_constants(top)))
end

"""
    _lattice_marker(lattice_kind)

Return the marker used by a general lattice display.
"""
_lattice_marker(::Type{<:LatticeType}) = :circle
_lattice_marker(::Type{Hexagonal}) = :hexagon

"""
    _lattice_marker_size(top)

Return the data-space marker diameter used for general lattice displays.
"""
function _lattice_marker_size(top::T) where {Kind,T<:LatticeTopology{Kind}}
    return Float32(_topology_marker_fill_factor(Kind) * minimum(lattice_constants(top)))
end

"""
    _topology_marker_fill_factor(::Type)

Return the data-space marker scale used to visually reduce gaps between lattice
markers. Hexagonal markers get a slightly larger fill factor because small gaps
between neighboring hex cells are visually more distracting than gaps between
round markers.
"""
_topology_marker_fill_factor(::Type{T}) where {T} = 1.04
_topology_marker_fill_factor(::Type{T}) where {T<:HexagonalTopology} = 1.12
_topology_marker_fill_factor(::Type{Hexagonal}) = 1.12

"""
    _topology_3d_marker_size(top, vals_size)

Return the mesh marker diameter for a three-dimensional topology display.
"""
function _topology_3d_marker_size(top::T, vals_size::NTuple{3,<:Integer}) where {T<:AbstractLayerTopology}
    return 0.3f0
end

function _topology_3d_marker_size(top::T, vals_size::NTuple{3,<:Integer}) where {U,P,T<:SquareTopology{U,3,P}}
    return 0.10f0
end

function _set_topology_display_colorrange!(plot, obs, layer::L, colorrange) where {L<:AbstractIsingLayer}
    if isnothing(colorrange)
        Windows._bind_layer_colorrange!(plot, obs, layer)
    else
        plot.colorrange[] = colorrange
    end
    return plot
end
