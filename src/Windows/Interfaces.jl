"""
    AxisTrait

Trait for panels that expose a meaningful Makie axis through `getaxis`.
Panel authors extend `axis_trait(::Type{<:MyPanel})`.
"""
abstract type AxisTrait end

"""
    ImageTrait

Trait for panels that can build a minimal, export-oriented Makie
representation with `toimage!`.

The image representation is intentionally separate from the mounted UI. A panel
can omit buttons, sliders, text boxes, and other interactive chrome while still
exporting the data view that is useful as an image.
"""
abstract type ImageTrait end

"""
    HasAxis()

Trait value for panels whose handle stores a meaningful Makie axis.
"""
struct HasAxis <: AxisTrait end

"""
    NoAxis()

Trait value for panels without a direct axis export.
"""
struct NoAxis <: AxisTrait end

"""
    HasImage()

Trait value for panels that implement `toimage!(cell, panel, handle; kwargs...)`.
"""
struct HasImage <: ImageTrait end

"""
    NoImage()

Trait value for panels without a custom minimal image builder. Such panels can
still be exported if they expose an axis or have imageable children.
"""
struct NoImage <: ImageTrait end

"""
    axis_trait(panel_or_panel_type) -> AxisTrait

Return whether a panel type hosts a meaningful Makie axis. The default is
`NoAxis()`. Extend this for custom panels:

```julia
InteractiveIsing.Windows.axis_trait(::Type{MyPanel}) = HasAxis()
```
"""
axis_trait(::Type{<:AbstractPanel}) = NoAxis()
axis_trait(panel::AbstractPanel) = axis_trait(typeof(panel))
axis_trait(handle::PanelHandle) = axis_trait(handle.panel)

"""
    image_trait(panel_or_panel_type) -> ImageTrait

Return whether a panel type implements a custom minimal image builder. The
default is `NoImage()`. Extend this for custom panels that want to control their
export representation:

```julia
InteractiveIsing.Windows.image_trait(::Type{MyPanel}) = HasImage()

function InteractiveIsing.Windows.toimage!(cell, panel::MyPanel, handle; kwargs...)
    ax = Axis(cell; title = "export")
    lines!(ax, handle[:x][], handle[:y][])
    return ax
end
```
"""
image_trait(::Type{<:AbstractPanel}) = NoImage()
image_trait(panel::AbstractPanel) = image_trait(typeof(panel))
image_trait(handle::PanelHandle) = image_trait(handle.panel)

"""
    topology_layer_display!(handle, cell, topology, vals, layer; kwargs...)

Draw an interactive two-dimensional layer display into `cell`. The Windows
framework calls this API, while concrete packages extend it for their topology
types.
"""
function topology_layer_display!(handle::PanelHandle, cell, top::T, vals, layer; kwargs...) where {T<:AbstractLayerTopology}
    throw(ArgumentError("No Windows layer display is registered for topology $(typeof(top))."))
end

"""
    axiskey(panel_or_panel_type) -> Symbol

Data key used by `getaxis` when `axis_trait(panel) isa HasAxis`. The default is
`:axis`; panels that store their useful axis elsewhere can override it.
"""
axiskey(::Type{<:AbstractPanel}) = :axis
axiskey(panel::AbstractPanel) = axiskey(typeof(panel))

"""
    hasaxis(handle) -> Bool

Return whether a mounted panel handle currently exposes a Makie axis.
"""
function hasaxis(handle::PanelHandle)
    axis_trait(handle) isa HasAxis || return false
    return haskey(handle, axiskey(handle.panel))
end

hasaxis(panel::AbstractPanel) = axis_trait(panel) isa HasAxis

"""
    hasimage(handle) -> Bool

Return whether `handle` has a minimal image representation. A handle is
imageable when its panel opts into `HasImage`, exposes an axis, or has at least
one imageable child.
"""
function hasimage(handle::PanelHandle)
    image_trait(handle) isa HasImage && return true
    hasaxis(handle) && return true
    return any(hasimage, values(handle.children))
end

hasimage(panel::AbstractPanel) = image_trait(panel) isa HasImage || hasaxis(panel)

"""
    getaxis(handle) -> Axis

Return the meaningful Makie axis for a mounted panel. Panel types opt in with
`axis_trait` and optionally `axiskey`.
"""
getaxis(handle::PanelHandle) = getaxis(axis_trait(handle), handle)

function getaxis(::HasAxis, handle::PanelHandle)
    key = axiskey(handle.panel)
    haskey(handle, key) || throw(ArgumentError("Panel $(typeof(handle.panel)) has no mounted axis at key $key."))
    return handle[key]
end

function getaxis(::NoAxis, handle::PanelHandle)
    throw(ArgumentError("Panel $(typeof(handle.panel)) does not declare an axis. Extend `axis_trait` or `toimage!`."))
end

"""
    axis_to_png(path, handle_or_axis; kwargs...) -> path

Save a panel axis, or an axis object directly, to a PNG. Keyword arguments are
forwarded to Makie's `save`.
"""
axis_to_png(path::AbstractString, handle::PanelHandle; kwargs...) =
    axis_to_png(path, getaxis(handle); kwargs...)

function axis_to_png(path::AbstractString, axis; kwargs...)
    save(path, axis.scene; kwargs...)
    return path
end

"""
    tofigure(handle_or_host; size = (900, 700), figure_kwargs = (;), kwargs...)

Build a fresh Makie `Figure` containing the minimal image representation of a
mounted panel or whole window. Keyword arguments after `figure_kwargs` are
forwarded to `toimage!`.
"""
function tofigure(handle::PanelHandle; size = (900, 700), figure_kwargs = (;), kwargs...)
    fig = Figure(; size, figure_kwargs...)
    toimage!(fig[1, 1], handle; kwargs...)
    return fig
end

function tofigure(host::WindowHost; size = size(host.figure.scene), figure_kwargs = (;), kwargs...)
    fig = Figure(; size, figure_kwargs...)
    toimage!(fig[1, 1], host; kwargs...)
    return fig
end

"""
    toimage!(cell, handle_or_host; kwargs...)
    toimage!(cell, panel, handle; kwargs...)

Build a minimal Makie representation into `cell`. Panel authors usually extend
the three-argument method for their panel type. The fallback exports a custom
panel axis when possible, otherwise it composes imageable children.
"""
toimage!(cell, handle::PanelHandle; kwargs...) = toimage!(cell, handle.panel, handle; kwargs...)
toimage!(cell, host::WindowHost; kwargs...) = _children_toimage!(cell, host.children; kwargs...)

function toimage!(cell, panel::AbstractPanel, handle::PanelHandle; kwargs...)
    if hasaxis(handle)
        return _axis_snapshot_toimage!(cell, handle; kwargs...)
    elseif any(hasimage, values(handle.children))
        return _children_toimage!(cell, handle.children; kwargs...)
    else
        throw(ArgumentError("Panel $(typeof(panel)) has no image representation. Extend `toimage!`, `axis_trait`, or mount imageable children."))
    end
end

function _children_toimage!(cell, children; kwargs...)
    image_children = [child for child in values(children) if hasimage(child)]
    isempty(image_children) && throw(ArgumentError("Window has no imageable panels."))

    grid = GridLayout(cell)
    for (idx, child) in enumerate(image_children)
        slot = _image_child_slot(child, idx)
        toimage!(grid[slot...], child; kwargs...)
    end
    return grid
end

function _image_child_slot(child::PanelHandle, idx)
    slot = child.slot
    if slot isa Tuple && all(x -> x isa Integer, slot)
        return slot
    else
        return (idx, 1)
    end
end

function _axis_snapshot_toimage!(cell, handle::PanelHandle; kwargs...)
    ax = Axis(cell, aspect = DataAspect())
    hidedecorations!(ax)
    hidespines!(ax)
    image!(ax, colorbuffer(getaxis(handle).scene), interpolate = false)
    return ax
end

"""
    toimage(path, handle_or_host; size, figure_kwargs, image_kwargs, kwargs...) -> path

Export the useful visual content of a mounted panel or whole window. This first
builds a fresh, minimal Makie representation with `tofigure`, then saves that
figure. Keyword arguments are forwarded to Makie's `save`; use
`image_kwargs = (; ...)` for keywords that should be forwarded to `toimage!`.

```julia
toimage("simulation.png", host)
toimage("panel.png", handle)
```
"""
function toimage(
    path::AbstractString,
    handle::PanelHandle;
    size = (900, 700),
    figure_kwargs = (;),
    image_kwargs = (;),
    kwargs...,
)
    save(path, tofigure(handle; size, figure_kwargs, image_kwargs...); kwargs...)
    return path
end

function toimage(
    path::AbstractString,
    host::WindowHost;
    size = size(host.figure.scene),
    figure_kwargs = (;),
    image_kwargs = (;),
    kwargs...,
)
    save(path, tofigure(host; size, figure_kwargs, image_kwargs...); kwargs...)
    return path
end

"""
    fullimage(path, host_or_handle; kwargs...) -> path

Save the current full window figure, including UI controls and layout chrome.
This is the literal snapshot counterpart to `toimage`, which builds a minimal
export figure.
"""
function fullimage(path::AbstractString, host::WindowHost; kwargs...)
    save(path, host.figure; kwargs...)
    return path
end

function fullimage(path::AbstractString, handle::PanelHandle; kwargs...)
    return fullimage(path, handle.host; kwargs...)
end
