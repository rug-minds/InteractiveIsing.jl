"""
    ConnectionsPanel(g; max_edges = 20_000, show_nodes = true,
                     selected_nodes = nothing, selection_mode = :incident,
                     curved = true, curve_resolution = 8, curve_amount = 0.15,
                     color_by_strength = true, colormap = :viridis,
                     colorrange = :auto, axis_kwargs = (;),
                     line_kwargs = (;), node_kwargs = (;))

Panel that visualizes graph adjacency entries as lines between lattice sites.

The panel derives node coordinates from the graph layer structure and reads
unique off-diagonal edges from `adj(g)`. A single two-dimensional layer is drawn
on an `Axis`; layered or three-dimensional graphs are drawn on an `Axis3`.
`max_edges` caps the number of rendered edges to keep large graphs responsive.
When `curved` is true, edges are drawn as short quadratic curves. When
`color_by_strength` is true, edge colors are mapped from the adjacency weight.

If `selected_nodes` is provided, only selected-node connections are shown.
`selected_nodes` may contain graph indices, or `(layer_index, coords)` tuples
where `coords` is a layer coordinate tuple. `selection_mode = :incident` shows
edges touching any selected node; `selection_mode = :within` shows only edges
between selected nodes.
"""
struct ConnectionsPanel <: AbstractPanel
    graph::Any
    max_edges::Int
    show_nodes::Bool
    selected_nodes::Any
    selection_mode::Symbol
    curved::Bool
    curve_resolution::Int
    curve_amount::Float32
    color_by_strength::Bool
    colormap::Any
    colorrange::Any
    axis_kwargs::Any
    line_kwargs::Any
    node_kwargs::Any
end

axis_trait(::Type{ConnectionsPanel}) = HasAxis()
image_trait(::Type{ConnectionsPanel}) = HasImage()

function ConnectionsPanel(
    g;
    max_edges = 20_000,
    show_nodes = true,
    selected_nodes = nothing,
    selection_mode = :incident,
    curved = true,
    curve_resolution = 8,
    curve_amount = 0.15,
    color_by_strength = true,
    colormap = :viridis,
    colorrange = :auto,
    axis_kwargs = (;),
    line_kwargs = (;),
    node_kwargs = (;),
)
    return ConnectionsPanel(
        g,
        max(0, Int(max_edges)),
        Bool(show_nodes),
        selected_nodes,
        Symbol(selection_mode),
        Bool(curved),
        max(2, Int(curve_resolution)),
        Float32(curve_amount),
        Bool(color_by_strength),
        colormap,
        colorrange,
        axis_kwargs,
        line_kwargs,
        node_kwargs,
    )
end

function mount!(panel::ConnectionsPanel, host::WindowHost, cell; kwargs...)
    grid = GridLayout(cell)
    handle = PanelHandle(panel, host, grid)

    plot_dim = _connection_plot_dimension(panel.graph)
    nodes = _connection_node_points(panel.graph, plot_dim)
    edge_points, edge_colors, edge_weights, total_edges =
        _connection_edge_geometry(panel.graph, nodes, panel)
    selected = _selected_graph_indices(panel.graph, panel.selected_nodes)

    handle[:node_points] = nodes
    handle[:selected_nodes] = selected
    handle[:edge_points] = Observable(edge_points)
    handle[:edge_colors] = Observable(edge_colors)
    handle[:edge_weights] = edge_weights
    handle[:edge_count] = total_edges
    handle[:visible_edge_count] = length(edge_weights)

    if plot_dim == 2
        ax = handle[:axis] = Axis(grid[1, 1]; _connection_axis_kwargs(panel, total_edges)...)
    else
        ax = handle[:axis] = Axis3(grid[1, 1]; _connection_axis_kwargs(panel, total_edges)...)
    end

    handle[:edge_plot] = lines!(ax, handle[:edge_points]; _connection_line_kwargs(panel, handle)...)
    if panel.show_nodes
        handle[:node_plot] = scatter!(ax, nodes; _connection_node_kwargs(panel)...)
        if !isnothing(selected) && !isempty(selected)
            handle[:selected_node_plot] = scatter!(ax, nodes[collect(selected)]; _selected_connection_node_kwargs(panel)...)
        end
    end
    return handle
end

function toimage!(cell, panel::ConnectionsPanel, handle::PanelHandle; kwargs...)
    total_edges = get(handle.data, :edge_count, 0)
    nodes = handle[:node_points]
    edge_points = handle[:edge_points][]

    if _connection_plot_dimension(panel.graph) == 2
        ax = Axis(cell; _connection_axis_kwargs(panel, total_edges)...)
    else
        ax = Axis3(cell; _connection_axis_kwargs(panel, total_edges)...)
        if haskey(handle, :axis)
            _restore_axis3_state!(ax, _axis3_state(handle[:axis]))
        end
    end

    lines!(ax, edge_points; _connection_line_kwargs(panel, handle)...)
    if panel.show_nodes
        scatter!(ax, nodes; _connection_node_kwargs(panel)...)
        selected = get(handle.data, :selected_nodes, nothing)
        if !isnothing(selected) && !isempty(selected)
            scatter!(ax, nodes[collect(selected)]; _selected_connection_node_kwargs(panel)...)
        end
    end
    return ax
end

function _connection_axis_kwargs(panel::ConnectionsPanel, total_edges)
    selected_text = isnothing(panel.selected_nodes) ? "" : ", selected"
    defaults = (;
        title = "Connections ($(min(total_edges, panel.max_edges))/$(total_edges)$(selected_text))",
    )
    return merge(defaults, panel.axis_kwargs)
end

function _connection_line_kwargs(panel::ConnectionsPanel, handle)
    defaults = (; linewidth = 1)
    kwargs = merge(defaults, panel.line_kwargs)
    if panel.color_by_strength
        kwargs = _without_namedtuple_keys(kwargs, (:color, :colormap, :colorrange))
        return merge(
            kwargs,
            (;
                color = handle[:edge_colors],
                colormap = panel.colormap,
                colorrange = _connection_colorrange(panel, handle[:edge_weights]),
            ),
        )
    else
        return merge((; color = (:black, 0.25)), kwargs)
    end
end

function _connection_node_kwargs(panel::ConnectionsPanel)
    defaults = (; color = (:dodgerblue, 0.7), markersize = 5)
    return merge(defaults, panel.node_kwargs)
end

function _selected_connection_node_kwargs(panel::ConnectionsPanel)
    defaults = (; color = :red, markersize = 11)
    return merge(defaults, panel.node_kwargs)
end

function _without_namedtuple_keys(nt::NamedTuple, keys)
    kept = Tuple(k for k in propertynames(nt) if k ∉ keys)
    return NamedTuple{kept}(map(k -> getproperty(nt, k), kept))
end

function _connection_colorrange(panel::ConnectionsPanel, weights)
    panel.colorrange === :auto || return panel.colorrange
    isempty(weights) && return (0.0f0, 1.0f0)

    lo, hi = extrema(Float32.(weights))
    if lo == hi
        δ = max(abs(lo), 1.0f0)
        return (lo - δ, hi + δ)
    end
    return (lo, hi)
end

function _connection_plot_dimension(g)
    graph_layers = layers(g)
    if length(graph_layers) == 1 && length(size(first(graph_layers))) <= 2
        return 2
    else
        return 3
    end
end

function _connection_node_points(g, ::Val{2})
    points = Vector{Point2f}(undef, nstates(g))
    for (layer_idx, layer) in enumerate(layers(g))
        _fill_connection_node_points!(points, layer, layer_idx, Val(2))
    end
    return points
end

function _connection_node_points(g, ::Val{3})
    points = Vector{Point3f}(undef, nstates(g))
    for (layer_idx, layer) in enumerate(layers(g))
        _fill_connection_node_points!(points, layer, layer_idx, Val(3))
    end
    return points
end

_connection_node_points(g, plot_dim::Integer) = _connection_node_points(g, Val(plot_dim))

function _fill_connection_node_points!(points, layer, layer_idx, ::Val{2})
    for (local_idx, graph_idx) in enumerate(graphidxs(layer))
        coords = idxToCoord(local_idx, size(layer))
        points[Int(graph_idx)] = Point2f(_coord_value(coords, 1), _coord_value(coords, 2))
    end
    return points
end

function _fill_connection_node_points!(points, layer, layer_idx, ::Val{3})
    for (local_idx, graph_idx) in enumerate(graphidxs(layer))
        coords = idxToCoord(local_idx, size(layer))
        points[Int(graph_idx)] = Point3f(
            _coord_value(coords, 1),
            _coord_value(coords, 2),
            length(coords) >= 3 ? _coord_value(coords, 3) : Float32(layer_idx),
        )
    end
    return points
end

_coord_value(coords, idx) = Float32(idx <= length(coords) ? coords[idx] : 1)

function _connection_edge_geometry(g, nodes::Vector{P}, panel::ConnectionsPanel) where {P}
    rows, cols, vals = SparseArrays.findnz(adj(g))
    selected = _selected_graph_indices(g, panel.selected_nodes)
    selected_nodes = isnothing(selected) ? nothing : Set(selected)
    edges = Tuple{Int,Int,eltype(vals)}[]
    edge_points = P[]
    edge_colors = Float32[]
    edge_weights = eltype(vals)[]
    seen = Set{Tuple{Int,Int}}()

    for k in eachindex(vals)
        row = Int(rows[k])
        col = Int(cols[k])
        row == col && continue

        a, b = minmax(row, col)
        edge = (a, b)
        edge in seen && continue
        push!(seen, edge)

        _show_connection_edge(a, b, selected_nodes, panel.selection_mode) || continue
        push!(edges, (a, b, vals[k]))
    end

    total_edges = length(edges)
    shown_edges = _sample_connection_edges!(edges, panel.max_edges)
    for (a, b, weight) in shown_edges
        _push_connection_edge!(edge_points, edge_colors, nodes[a], nodes[b], weight, (a, b), panel)
        push!(edge_weights, weight)
    end
    return edge_points, edge_colors, edge_weights, total_edges
end

function _show_connection_edge(a, b, selected_nodes::Nothing, selection_mode)
    return true
end

function _show_connection_edge(a, b, selected_nodes::Set, selection_mode)
    if selection_mode === :incident
        return a in selected_nodes || b in selected_nodes
    elseif selection_mode === :within
        return a in selected_nodes && b in selected_nodes
    else
        throw(ArgumentError("Unknown connection selection_mode $(selection_mode). Use :incident or :within."))
    end
end

function _sample_connection_edges!(edges, max_edges)
    if length(edges) <= max_edges
        return edges
    end

    sort!(edges, by = edge -> hash((edge[1], edge[2])))
    return view(edges, 1:max_edges)
end

_selected_graph_indices(g, ::Nothing) = nothing
_selected_graph_indices(g, selected_node::Integer) =
    _clean_selected_graph_indices(g, [_selected_graph_index(g, selected_node)])
_selected_graph_indices(g, selected_node::Tuple{<:Integer,<:Tuple}) =
    _clean_selected_graph_indices(g, [_selected_graph_index(g, selected_node)])

function _selected_graph_indices(g, selected_nodes)
    selected = Int[]
    for node in selected_nodes
        push!(selected, _selected_graph_index(g, node))
    end
    return _clean_selected_graph_indices(g, selected)
end

function _clean_selected_graph_indices(g, selected)
    filter!(idx -> 1 <= idx <= nstates(g), selected)
    return unique(selected)
end

_selected_graph_index(g, idx::Integer) = Int(idx)

function _selected_graph_index(g, spec::Tuple{<:Integer,<:Tuple})
    layer_idx = Int(spec[1])
    coords = spec[2]
    layer = layers(g)[layer_idx]
    local_idx = LinearIndices(size(layer))[coords...]
    return Int(graphidxs(layer)[local_idx])
end

function _push_connection_edge!(points, colors, a::Point2f, b::Point2f, weight, edge, panel)
    if panel.curved
        _push_curved_connection_edge!(points, colors, a, b, weight, edge, panel)
    else
        push!(points, a, b, Point2f(NaN32, NaN32))
        _push_edge_colors!(colors, weight, 3)
    end
    return points
end

function _push_connection_edge!(points, colors, a::Point3f, b::Point3f, weight, edge, panel)
    if panel.curved
        _push_curved_connection_edge!(points, colors, a, b, weight, edge, panel)
    else
        push!(points, a, b, Point3f(NaN32, NaN32, NaN32))
        _push_edge_colors!(colors, weight, 3)
    end
    return points
end

function _push_curved_connection_edge!(points, colors, a::Point2f, b::Point2f, weight, edge, panel)
    x1, y1 = a[1], a[2]
    x2, y2 = b[1], b[2]
    dx, dy = x2 - x1, y2 - y1
    len = sqrt(dx * dx + dy * dy)
    invlen = len > 0 ? inv(len) : 0.0f0
    sign = _connection_curve_sign(edge)
    offset = sign * panel.curve_amount * max(len, 1.0f0)
    cx = (x1 + x2) / 2 - dy * invlen * offset
    cy = (y1 + y2) / 2 + dx * invlen * offset

    for i in 0:(panel.curve_resolution - 1)
        t = Float32(i / (panel.curve_resolution - 1))
        omt = 1 - t
        push!(points, Point2f(omt * omt * x1 + 2 * omt * t * cx + t * t * x2,
                              omt * omt * y1 + 2 * omt * t * cy + t * t * y2))
    end
    push!(points, Point2f(NaN32, NaN32))
    _push_edge_colors!(colors, weight, panel.curve_resolution + 1)
    return points
end

function _push_curved_connection_edge!(points, colors, a::Point3f, b::Point3f, weight, edge, panel)
    x1, y1, z1 = a[1], a[2], a[3]
    x2, y2, z2 = b[1], b[2], b[3]
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    len = sqrt(dx * dx + dy * dy + dz * dz)
    invlen = len > 0 ? inv(len) : 0.0f0
    sign = _connection_curve_sign(edge)
    px, py, pz = -dy, dx, 0.0f0
    plen = sqrt(px * px + py * py + pz * pz)
    if plen <= 0
        px, py, pz = 0.0f0, -dz, dy
        plen = sqrt(px * px + py * py + pz * pz)
    end
    poffset = sign * panel.curve_amount * max(len, 1.0f0) / max(plen, 1.0f0)
    lift = 0.35f0 * panel.curve_amount * max(len, 1.0f0)
    cx = (x1 + x2) / 2 + px * poffset
    cy = (y1 + y2) / 2 + py * poffset
    cz = (z1 + z2) / 2 + pz * poffset + lift

    for i in 0:(panel.curve_resolution - 1)
        t = Float32(i / (panel.curve_resolution - 1))
        omt = 1 - t
        push!(points, Point3f(omt * omt * x1 + 2 * omt * t * cx + t * t * x2,
                              omt * omt * y1 + 2 * omt * t * cy + t * t * y2,
                              omt * omt * z1 + 2 * omt * t * cz + t * t * z2))
    end
    push!(points, Point3f(NaN32, NaN32, NaN32))
    _push_edge_colors!(colors, weight, panel.curve_resolution + 1)
    return points
end

_connection_curve_sign(edge) = isodd(hash(edge)) ? 1.0f0 : -1.0f0

function _push_edge_colors!(colors, weight, n)
    append!(colors, Iterators.repeated(Float32(weight), n))
    return colors
end
