"""
    _defect_bound_layer(defects)

Return the graph layer owned by a bound defect-hopping proposer.
"""
function _defect_bound_layer(defects::D) where {D<:DefectHopping}
    isnothing(defects.state) &&
        throw(ArgumentError("DefectHopping must be bound to an IsingGraph before calling interface(defects)."))
    return defects.state[Int(defects.layer)]
end

function _defect_bound_layer(charges::C) where {C<:NeutralChargeHopping}
    return _defect_bound_layer(charges.positive)
end

"""
    _defect_world_point(layer, graph_idx)

Convert one graph index into a three-dimensional Makie point.
"""
function _defect_world_point(layer::L, graph_idx::I) where {L<:AbstractIsingLayer,I<:Integer}
    local_idx = Int(graph_idx - startidx(layer) + 1)
    ci = CartesianIndices(size(layer))[local_idx]
    wc = woorldcoordinate(topology(layer), Coordinate(topology(layer), ci; check = false))

    # Makie can render all defect systems with Axis3 by padding 1D/2D layers.
    x = Float32(wc[1])
    y = length(wc) >= 2 ? Float32(wc[2]) : 0f0
    z = length(wc) >= 3 ? Float32(wc[3]) : 0f0
    return Point3f(x, y, z)
end

"""
    _defect_display_points(defects)

Return the current defect marker positions for Makie display.
"""
function _defect_display_points(defects::D) where {D<:DefectHopping}
    layer = _defect_bound_layer(defects)
    points = Vector{Point3f}(undef, length(defects.defect_idxs))
    for i in eachindex(defects.defect_idxs)
        @inbounds points[i] = _defect_world_point(layer, defects.defect_idxs[i])
    end
    return points
end

function _defect_positive_display_points(charges::C) where {C<:NeutralChargeHopping}
    return _defect_display_points(charges.positive)
end

function _defect_negative_display_points(charges::C) where {C<:NeutralChargeHopping}
    return _defect_display_points(charges.negative)
end

"""
    _defect_lattice_points(defects)

Return one faint background point for every site in the defect layer.
"""
function _defect_lattice_points(defects::D) where {D<:DefectHopping}
    layer = _defect_bound_layer(defects)
    points = Vector{Point3f}(undef, nStates(layer))
    for local_idx in 1:nStates(layer)
        graph_idx = _defect_graph_index(layer, local_idx)
        @inbounds points[local_idx] = _defect_world_point(layer, graph_idx)
    end
    return points
end

"""
    Windows.interface(defects::DefectHopping; kwargs...)

Open a live GLMakie display for the bound defect-hopping proposer. The marker
positions follow accepted Metropolis defect hops by polling the proposer's
current `defect_idxs`.
"""
function Windows.interface(
    defects::D;
    framerate = 30,
    polling_rate = 10,
    size = (900, 800),
    title = "Defect hopping",
    defect_markersize = 0.42,
    lattice_markersize = 0.12,
    defect_color = :deepskyblue,
    lattice_color = (:gray70, 0.045),
    show_lattice = true,
    focus = true,
) where {D<:DefectHopping}
    host = Windows.window(; title, size, fps = framerate, polling_rate, focus)
    ax = host[:axis] = Axis3(host.figure[1, 1])
    ax.xlabel = "x"
    ax.ylabel = "y"
    ax.zlabel = "z"

    if show_lattice
        host[:lattice_plot] = meshscatter!(
            ax,
            _defect_lattice_points(defects);
            markersize = lattice_markersize,
            color = lattice_color,
        )
    end

    points = Windows.hot_observable!(host, _defect_display_points(defects))
    host[:defect_points] = points
    host[:defect_plot] = meshscatter!(
        ax,
        points;
        markersize = defect_markersize,
        color = defect_color,
    )
    host[:defects] = defects

    Windows.register_frame!(host) do _
        points[] = _defect_display_points(defects)
        return nothing
    end

    return host
end

"""
    Windows.interface(charges::NeutralChargeHopping; kwargs...)

Open one live GLMakie display for a neutral charge model, with positive and
negative mobile charges shown in different colors.
"""
function Windows.interface(
    charges::C;
    framerate = 30,
    polling_rate = 10,
    size = (900, 800),
    title = "Neutral charge hopping",
    charge_markersize = 0.42,
    positive_markersize = charge_markersize,
    negative_markersize = 0.7 * charge_markersize,
    lattice_markersize = 0.12,
    positive_color = :red,
    negative_color = :cyan,
    lattice_color = (:gray70, 0.045),
    show_lattice = true,
    focus = true,
) where {C<:NeutralChargeHopping}
    host = Windows.window(; title, size, fps = framerate, polling_rate, focus)
    ax = host[:axis] = Axis3(host.figure[1, 1])
    ax.xlabel = "x"
    ax.ylabel = "y"
    ax.zlabel = "z"

    if show_lattice
        host[:lattice_plot] = meshscatter!(
            ax,
            _defect_lattice_points(charges.positive);
            markersize = lattice_markersize,
            color = lattice_color,
        )
    end

    positive_points = Windows.hot_observable!(host, _defect_positive_display_points(charges))
    negative_points = Windows.hot_observable!(host, _defect_negative_display_points(charges))
    host[:positive_points] = positive_points
    host[:negative_points] = negative_points
    host[:positive_plot] = meshscatter!(
        ax,
        positive_points;
        markersize = positive_markersize,
        color = positive_color,
    )
    host[:negative_plot] = meshscatter!(
        ax,
        negative_points;
        markersize = negative_markersize,
        color = negative_color,
    )
    host[:charges] = charges

    Windows.register_frame!(host) do _
        positive_points[] = _defect_positive_display_points(charges)
        negative_points[] = _defect_negative_display_points(charges)
        return nothing
    end

    return host
end
