struct LineSnapshot
    x_source::Any
    y_source::Any
    x_view::Any
    y_view::Any
    nonce::UInt
end

Base.:(==)(a::LineSnapshot, b::LineSnapshot) = a.nonce == b.nonce

"""
    InteractiveLinesPanel(x, y; kwargs...)
    InteractiveLinesPanel(getter; kwargs...)

Panel that plots two dynamically updated line containers.

The simple constructor accepts two vector-like containers. The getter
constructor accepts a zero-argument function returning `(x, y)`, so callers can
compute or mutate owned buffers before returning the current containers:

```julia
xs = Float64[]
ys = Float64[]
InteractiveLinesPanel(() -> begin
    push!(xs, time())
    push!(ys, norm(matrix))
    return xs, ys
end)
```

The plot observables hold prefix views into the returned containers. On each
poll, the views are replaced with fresh zero-copy views of length
`min(length(x), length(y))`, so Makie always sees matched x/y lengths while data
stays in the source containers.
"""
struct InteractiveLinesPanel <: AbstractPanel
    getter::Any
    axis_kwargs::Any
    line_kwargs::Any
    autolimits::Bool
    xlabel::String
    ylabel::String
    title::String
    update_rate::Float64
end

axis_trait(::Type{InteractiveLinesPanel}) = HasAxis()
image_trait(::Type{InteractiveLinesPanel}) = HasImage()

function InteractiveLinesPanel(
    getter::Function;
    xlabel = "",
    ylabel = "",
    title = "",
    axis_kwargs = (;),
    line_kwargs = (;),
    autolimits = true,
    update_rate = 10,
)
    return InteractiveLinesPanel(
        getter,
        axis_kwargs,
        line_kwargs,
        Bool(autolimits),
        string(xlabel),
        string(ylabel),
        string(title),
        Float64(update_rate),
    )
end

InteractiveLinesPanel(x, y; kwargs...) = InteractiveLinesPanel(() -> (x, y); kwargs...)

"""
    ContextLinesPanel(context, xvar, yvar; kwargs...)

Alternate constructor for `InteractiveLinesPanel` that maps process context
variables to the generic interactive line source.

`context` may be a `Processes.ProcessContext`, a `Processes.AbstractProcess`, or
a zero-argument function returning either of those. Supported variable selectors
are:

- `Processes.Var(:subcontext, :name)`
- `:subcontext => :name`
- `algorithm => :name`, where `algorithm` is a process algorithm object used
  in the context
- `(:subcontext, :name)`
- `:name` for globals or a top-level property
"""
function ContextLinesPanel(context, xvar, yvar; kwargs...)
    return InteractiveLinesPanel(_context_var_value(context, xvar), _context_var_value(context, yvar); kwargs...)
end

function mount!(panel::InteractiveLinesPanel, host::WindowHost, cell; kwargs...)
    grid = GridLayout(cell)
    handle = PanelHandle(panel, host, grid)

    ax = handle[:axis] = Axis(grid[1, 1]; _interactive_lines_axis_kwargs(panel)...)
    snapshot = _line_snapshot(panel.getter)
    handle[:x_container] = snapshot.x_source
    handle[:y_container] = snapshot.y_source
    handle[:x_obs] = Observable{Any}(snapshot.x_view)
    handle[:y_obs] = Observable{Any}(snapshot.y_view)
    handle[:last_update_time] = 0.0
    handle[:update_interval] = panel.update_rate <= 0 ? 0.0 : 1 / panel.update_rate
    handle[:plot] = lines!(ax, handle[:x_obs], handle[:y_obs]; panel.line_kwargs...)

    poller = handle[:line_snapshot] = register_polled!(
        handle,
        PolledObservable(snapshot, po -> _poll_interactive_lines!(handle, po)),
    )
    register!(handle, on(poller) do snap
        _set_interactive_line_snapshot!(handle, snap)
    end)
    return handle
end

function _interactive_lines_axis_kwargs(panel::InteractiveLinesPanel)
    axis_kwargs = panel.axis_kwargs
    isempty(panel.xlabel) || (axis_kwargs = merge(axis_kwargs, (; xlabel = panel.xlabel)))
    isempty(panel.ylabel) || (axis_kwargs = merge(axis_kwargs, (; ylabel = panel.ylabel)))
    isempty(panel.title) || (axis_kwargs = merge(axis_kwargs, (; title = panel.title)))
    return axis_kwargs
end

function _context_source(source::Function)
    return _context_source(source())
end
_context_source(process::Processes.AbstractProcess) = getfield(process, :context)
_context_source(context) = context

function _context_var_value(source, var)
    context = _context_source(source)
    return _context_var_value_from_context(context, var)
end

_context_var_value_from_context(context, var::Processes.Var) = context[var]
_context_var_value_from_context(context, var::Pair{Symbol, Symbol}) =
    getproperty(getproperty(context, first(var)), last(var))
_context_var_value_from_context(context, var::Pair) =
    getproperty(context[first(var)], last(var))
_context_var_value_from_context(context, var::Tuple{Symbol, Symbol}) =
    getproperty(getproperty(context, first(var)), last(var))

function _context_var_value_from_context(context::Processes.ProcessContext, var::Symbol)
    globals = Processes.getglobals(context)
    haskey(globals, var) && return getproperty(globals, var)
    return getproperty(context, var)
end

_context_var_value_from_context(context, var::Symbol) = getproperty(context, var)

function _line_snapshot(getter)
    x, y = getter()
    xview, yview = _matched_line_views(x, y)
    _validate_line_views(xview, yview)
    return LineSnapshot(x, y, xview, yview, UInt(time_ns()))
end

function _matched_line_views(x, y)
    n = min(_container_length(x), _container_length(y))
    return _container_prefix_view(x, n), _container_prefix_view(y, n)
end

function _validate_line_views(xview, yview)
    if isnothing(xview) || isnothing(yview)
        error("InteractiveLinesPanel currently requires viewable vector-like containers.")
    end
    return nothing
end

_container_length(container) = length(container)

_container_prefix_view(container, n::Integer) = nothing

function _container_prefix_view(container::AbstractVector, n::Integer)
    lo = firstindex(container)
    n <= 0 && return view(container, lo:lo-1)
    hi = lo + n - 1
    return view(container, lo:hi)
end

function _poll_interactive_lines!(handle::PanelHandle, po)
    haskey(handle, :x_obs) || return po[]
    haskey(handle, :y_obs) || return po[]
    now = time()
    if now - handle[:last_update_time] < handle[:update_interval]
        return po[]
    end
    handle[:last_update_time] = now
    return _line_snapshot(handle.panel.getter)
end

function _set_interactive_line_snapshot!(handle::PanelHandle, snapshot::LineSnapshot)
    handle[:x_container] = snapshot.x_source
    handle[:y_container] = snapshot.y_source
    handle[:x_obs].val = snapshot.x_view
    handle[:y_obs].val = snapshot.y_view
    notify(handle[:x_obs])
    notify(handle[:y_obs])

    if getfield(handle.panel, :autolimits) && haskey(handle, :axis)
        autolimits!(handle[:axis])
    end
    return nothing
end

function toimage!(cell, panel::InteractiveLinesPanel, handle::PanelHandle; kwargs...)
    ax = Axis(cell; _interactive_lines_axis_kwargs(panel)...)
    snapshot = haskey(handle, :x_obs) && haskey(handle, :y_obs) ? nothing : _line_snapshot(panel.getter)
    x = haskey(handle, :x_obs) ? handle[:x_obs][] : snapshot.x_view
    y = haskey(handle, :y_obs) ? handle[:y_obs][] : snapshot.y_view
    lines!(ax, x, y; panel.line_kwargs...)
    panel.autolimits && autolimits!(ax)
    return ax
end
