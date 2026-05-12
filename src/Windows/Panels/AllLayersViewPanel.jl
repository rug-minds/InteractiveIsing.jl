"""
    AllLayersViewPanel(g; colormap = :thermal, labels = true,
                       initial_view = :all, axis_kwargs = (;))

Display all positioned 2D layers of `g` in one shared Makie axis.

Layer coordinates are read from `coords(layer)`, interpreted as `(y, x, z)`,
and only the xy position is used. The xy coordinate is treated as the global
lower-left origin of that layer's image rectangle. Layers must be 2D and must
have unique xy coordinates; duplicate coordinates or overlapping rectangles
throw an `ArgumentError`.

The resulting axis keeps Makie's normal drag-pan and scroll-zoom interactions,
so large layer arrangements can be inspected as one continuous view.
"""
struct AllLayersViewPanel <: AbstractPanel
    graph::Any
    colormap::Any
    labels::Bool
    initial_view::Symbol
    axis_kwargs::Any
end

axis_trait(::Type{AllLayersViewPanel}) = HasAxis()
image_trait(::Type{AllLayersViewPanel}) = HasImage()

function AllLayersViewPanel(g; colormap = :thermal, labels = true, initial_view = :all, axis_kwargs = (;))
    return AllLayersViewPanel(g, colormap, Bool(labels), Symbol(initial_view), axis_kwargs)
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
    placements = _all_layer_placements(panel.graph)
    ax = handle[:axis] = Axis(cell; _all_layers_axis_kwargs(panel)...)
    ax.yreversed = @load_preference("makie_y_flip", default = false)
    handle[:placements] = placements
    handle[:layer_observables] = Observable[]
    handle[:plots] = Any[]

    for placement in placements
        obs = Observable(state(placement.layer))
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
                color = :black,
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

function _all_layer_placements(g)
    graph_layers = collect(layers(g))
    isempty(graph_layers) && throw(ArgumentError("AllLayersViewPanel needs at least one layer."))
    all(layer -> layer isa AbstractIsingLayer{<:Any,2}, graph_layers) ||
        throw(ArgumentError("AllLayersViewPanel currently supports only graphs made of 2D layers."))

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

        height = Float32(size(layer, 1))
        width = Float32(size(layer, 2))
        x0 = Float32(c[2])
        y0 = Float32(c[1])
        push!(placements, AllLayerPlacement(idx, layer, xy, x0, x0 + width, y0, y0 + height))
    end

    _assert_no_layer_rect_overlap!(placements)
    return placements
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
