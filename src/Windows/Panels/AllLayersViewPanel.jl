"""
    AllLayersViewPanel(g; colormap = :thermal, labels = true,
                       initial_view = :all, axis_kwargs = (;),
                       display_sizes = nothing)

Display all positioned 1D and 2D layers of `g` in one shared Makie axis.

Layer coordinates are read from `coords(layer)`, interpreted as `(y, x, z)`,
and only the xy position is used. The xy coordinate is treated as the global
lower-left origin of that layer's image rectangle. One-dimensional layers are
drawn as compact 2D grids. Layers must have unique xy coordinates; duplicate
coordinates throw an `ArgumentError`.

The resulting axis keeps Makie's normal drag-pan and scroll-zoom interactions,
so large layer arrangements can be inspected as one continuous view.

`display_sizes` may be a tuple/vector/dict of `(height, width)` rectangles by
layer index. It changes only the rendered rectangle size, not the graph layer
or state size.
"""
struct AllLayersViewPanel <: AbstractPanel
    graph::Any
    colormap::Any
    labels::Bool
    initial_view::Symbol
    axis_kwargs::Any
    display_sizes::Any
end

axis_trait(::Type{AllLayersViewPanel}) = HasAxis()
image_trait(::Type{AllLayersViewPanel}) = HasImage()

function AllLayersViewPanel(g; colormap = :thermal, labels = true, initial_view = :all, axis_kwargs = (;), display_sizes = nothing)
    return AllLayersViewPanel(g, colormap, Bool(labels), Symbol(initial_view), axis_kwargs, display_sizes)
end

function mount!(panel::AllLayersViewPanel, host::WindowHost, cell; kwargs...)
    grid = GridLayout(cell)
    handle = PanelHandle(panel, host, grid)
    _register_graph_close!(handle, panel.graph)
    _draw_all_layers_view!(handle, grid[1, 1])
    register_frame!(handle) do _
        for obs in get(handle.data, :layer_observables, Observable[])
            notify(obs)
        end
        return nothing
    end
    return handle
end

function toimage!(cell, panel::AllLayersViewPanel, handle::PanelHandle; kwargs...)
    export_handle = PanelHandle(panel, handle.host, cell)
    return _draw_all_layers_view!(export_handle, cell)
end

function _draw_all_layers_view!(handle::PanelHandle, cell)
    panel = handle.panel::AllLayersViewPanel
    placements = _all_layer_placements(panel.graph, panel.display_sizes)
    ax = handle[:axis] = Axis(cell; _all_layers_axis_kwargs(panel)...)
    ax.yreversed = @load_preference("makie_y_flip", default = true)
    handle[:placements] = placements
    handle[:layer_observables] = Observable[]
    handle[:plots] = Any[]

    for placement in placements
        obs = hot_observable!(handle, _all_layer_image_state(placement.layer))
        push!(handle[:layer_observables], obs)
        plot = image!(
            ax,
            placement.x0..placement.x1,
            placement.y0..placement.y1,
            obs;
            colormap = panel.colormap,
            fxaa = false,
            interpolate = false,
        )
        _bind_layer_colorrange!(plot, obs, placement.layer)
        push!(handle[:plots], plot)

        if panel.labels
            text!(
                ax,
                placement.x0 + 0.5f0,
                placement.y1 - 0.5f0;
                text = "Layer $(placement.index)",
                align = (:left, :top),
                fontsize = 12,
                color = :white,
            )
        end
    end

    _set_all_layers_limits!(ax, placements, panel.initial_view)
    return ax
end

function _all_layers_axis_kwargs(panel::AllLayersViewPanel)
    defaults = (;
        aspect = DataAspect(),
        title = "All layers",
        xlabel = "x",
        ylabel = "y",
    )
    return merge(defaults, panel.axis_kwargs)
end

struct AllLayerPlacement
    index::Int
    layer::Any
    xy::Tuple{Int32, Int32}
    x0::Float32
    x1::Float32
    y0::Float32
    y1::Float32
end

const _ALL_LAYER_AUTO_GAP = 2f0

struct _LayerVectorGrid{V<:AbstractVector} <: AbstractMatrix{eltype(V)}
    data::V
    size::Tuple{Int, Int}
end

Base.size(view::_LayerVectorGrid) = view.size
function Base.getindex(view::_LayerVectorGrid, i::Int, j::Int)
    return @inbounds view.data[i + (j - 1) * view.size[1]]
end
Base.IndexStyle(::Type{<:_LayerVectorGrid}) = IndexCartesian()

"""
    hot_observable_zero(::Type{<:_LayerVectorGrid})

Build the close-time inert replacement for a one-dimensional layer grid view
while preserving the concrete observable value type.
"""
function hot_observable_zero(::Type{T}) where {V,T<:_LayerVectorGrid{V}}
    data = hot_observable_zero(V)
    replacement = _LayerVectorGrid(data, (0, 0))
    replacement isa T && return replacement
    throw(ArgumentError("Cannot build zero-sized replacement for hot observable value type $T."))
end

_all_layer_image_state(layer::AbstractIsingLayer{<:Any,1}) =
    _LayerVectorGrid(vec(state(layer)), _all_layer_vector_grid_size(layer))

_all_layer_image_state(layer::AbstractIsingLayer{<:Any,2}) = state(layer)

function _all_layer_vector_grid_size(layer::AbstractIsingLayer{<:Any,1})
    n = prod(size(layer))
    height = floor(Int, sqrt(n))
    while height > 1 && n % height != 0
        height -= 1
    end
    width = cld(n, height)
    return (height, width)
end

function _all_layer_rect_size(layer::AbstractIsingLayer{<:Any,1})
    height, width = _all_layer_vector_grid_size(layer)
    return Float32(height), Float32(width)
end

function _all_layer_rect_size(layer::AbstractIsingLayer{<:Any,2})
    return Float32(size(layer, 1)), Float32(size(layer, 2))
end

function _all_layer_display_size(display_sizes, idx, layer)
    isnothing(display_sizes) && return _all_layer_rect_size(layer)
    value =
        display_sizes isa AbstractDict ? get(display_sizes, idx, nothing) :
        idx <= length(display_sizes) ? display_sizes[idx] :
        nothing
    isnothing(value) && return _all_layer_rect_size(layer)
    length(value) == 2 || throw(ArgumentError("display size for layer $idx must be (height, width), got $value"))
    return Float32(value[1]), Float32(value[2])
end

_supported_all_layer(layer) = layer isa AbstractIsingLayer{<:Any,1} || layer isa AbstractIsingLayer{<:Any,2}

function _all_layer_placements(g, display_sizes = nothing)
    graph_layers = collect(layers(g))
    isempty(graph_layers) && throw(ArgumentError("AllLayersViewPanel needs at least one layer."))
    all(_supported_all_layer, graph_layers) ||
        throw(ArgumentError("AllLayersViewPanel currently supports only graphs made of 1D or 2D layers."))

    placements = AllLayerPlacement[]
    seen_xy = Dict{Tuple{Int32,Int32}, Int}()
    for (idx, layer) in enumerate(graph_layers)
        c = coords(layer)
        isnothing(c) && throw(ArgumentError("Layer $idx has no coordinates. Set layer coords before using AllLayersViewPanel."))
        length(c) >= 2 || throw(ArgumentError("Layer $idx coordinates must contain at least y and x, got $c."))

        xy = (Int32(c[2]), Int32(c[1]))
        if haskey(seen_xy, xy)
            throw(ArgumentError("Layers $(seen_xy[xy]) and $idx share xy coordinate $xy. AllLayersViewPanel needs unique layer positions."))
        end
        seen_xy[xy] = idx

        height, width = _all_layer_display_size(display_sizes, idx, layer)
        x0 = Float32(c[2])
        y0 = Float32(c[1])
        push!(placements, AllLayerPlacement(idx, layer, xy, x0, x0 + width, y0, y0 + height))
    end

    if _any_layer_rect_overlap(placements)
        all(layer -> layer isa AbstractIsingLayer{<:Any,1}, graph_layers) ||
            _assert_no_layer_rect_overlap!(placements)
        placements = _auto_pack_1d_layer_placements(placements)
    end
    return placements
end

function _any_layer_rect_overlap(placements)
    for i in eachindex(placements)
        for j in (i + 1):lastindex(placements)
            _rects_overlap(placements[i], placements[j]) && return true
        end
    end
    return false
end

function _auto_pack_1d_layer_placements(placements)
    packed = AllLayerPlacement[]
    x0 = 0f0
    for placement in placements
        height = placement.y1 - placement.y0
        width = placement.x1 - placement.x0
        push!(
            packed,
            AllLayerPlacement(
                placement.index,
                placement.layer,
                (Int32(round(x0)), 0),
                x0,
                x0 + width,
                0f0,
                height,
            ),
        )
        x0 += width + _ALL_LAYER_AUTO_GAP
    end
    return packed
end

function _assert_no_layer_rect_overlap!(placements)
    for i in eachindex(placements)
        for j in (i + 1):lastindex(placements)
            _rects_overlap(placements[i], placements[j]) || continue
            throw(ArgumentError("Layers $(placements[i].index) and $(placements[j].index) overlap in global xy space."))
        end
    end
    return nothing
end

function _rects_overlap(a::AllLayerPlacement, b::AllLayerPlacement)
    return a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1
end

function _set_all_layers_limits!(ax, placements, initial_view::Symbol)
    if initial_view === :all
        xlo = minimum(p -> p.x0, placements)
        xhi = maximum(p -> p.x1, placements)
        ylo = minimum(p -> p.y0, placements)
        yhi = maximum(p -> p.y1, placements)
        limits!(ax, xlo, xhi, ylo, yhi)
    elseif initial_view === :first
        p = first(placements)
        limits!(ax, p.x0, p.x1, p.y0, p.y1)
    elseif initial_view === :auto
        autolimits!(ax)
    else
        throw(ArgumentError("Unknown AllLayersViewPanel initial_view $(initial_view). Use :all, :first, or :auto."))
    end
    return ax
end
