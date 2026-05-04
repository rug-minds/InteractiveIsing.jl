"""
    ContextLinesPanel(context, xvar, yvar; xlabel = "", ylabel = "", title = "",
                      axis_kwargs = (;), line_kwargs = (;), autolimits = true,
                      update_rate = 10)

Panel that plots two dynamically updated context containers as an interactive
line plot.

`context` may be a `Processes.ProcessContext`, a `Processes.AbstractProcess`, or
a zero-argument function returning either of those. `xvar` and `yvar` identify
the context variables during panel construction. After mounting, the panel keeps
direct references to the selected containers and no longer traverses the context
on every update. Supported variable selectors are:

- `Processes.Var(:subcontext, :name)`
- `:subcontext => :name`
- `algorithm => :name`, where `algorithm` is a process algorithm object used
  in the context
- `(:subcontext, :name)`
- `:name` for globals or a top-level property

The plot observables hold prefix views into those containers. On each update,
the views are replaced with fresh zero-copy views of length
`min(length(x), length(y))`, so Makie always sees matched x/y lengths while the
data still lives in the original containers.
"""
struct ContextLinesPanel{C, X, Y, A, L} <: AbstractPanel
    context::C
    xvar::X
    yvar::Y
    axis_kwargs::A
    line_kwargs::L
    autolimits::Bool
    xlabel::String
    ylabel::String
    title::String
    update_rate::Float64
end

function ContextLinesPanel(
    context,
    xvar,
    yvar;
    xlabel = "",
    ylabel = "",
    title = "",
    axis_kwargs = (;),
    line_kwargs = (;),
    autolimits = true,
    update_rate = 10,
)
    return ContextLinesPanel(
        context,
        xvar,
        yvar,
        axis_kwargs,
        line_kwargs,
        Bool(autolimits),
        string(xlabel),
        string(ylabel),
        string(title),
        Float64(update_rate),
    )
end

function mount!(panel::ContextLinesPanel, host::WindowHost, cell; kwargs...)
    grid = GridLayout(cell)
    handle = PanelHandle(panel, host, grid)

    ax = handle[:axis] = Axis(grid[1, 1]; _context_lines_axis_kwargs(panel)...)
    xcontainer, ycontainer = _context_line_containers(panel)
    handle[:x_container] = xcontainer
    handle[:y_container] = ycontainer
    xview, yview = _matched_container_views(xcontainer, ycontainer)
    _validate_context_line_views(xview, yview)
    xobs = handle[:x_obs] = Observable{Any}(xview)
    yobs = handle[:y_obs] = Observable{Any}(yview)
    handle[:last_update_time] = 0.0
    handle[:update_interval] = panel.update_rate <= 0 ? 0.0 : 1 / panel.update_rate
    handle[:plot] = lines!(ax, xobs, yobs; panel.line_kwargs...)

    register_frame!(handle) do _
        _refresh_context_lines!(handle)
    end
    return handle
end

function _context_lines_axis_kwargs(panel::ContextLinesPanel)
    axis_kwargs = panel.axis_kwargs
    isempty(panel.xlabel) || (axis_kwargs = merge(axis_kwargs, (; xlabel = panel.xlabel)))
    isempty(panel.ylabel) || (axis_kwargs = merge(axis_kwargs, (; ylabel = panel.ylabel)))
    isempty(panel.title) || (axis_kwargs = merge(axis_kwargs, (; title = panel.title)))
    return axis_kwargs
end

_context_source(source::Function) = _context_source(source())
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

function _context_line_containers(panel::ContextLinesPanel)
    x = _context_var_value(panel.context, panel.xvar)
    y = _context_var_value(panel.context, panel.yvar)
    return x, y
end

function _matched_container_views(x, y)
    n = min(_container_length(x), _container_length(y))
    return _container_prefix_view(x, n), _container_prefix_view(y, n)
end

function _validate_context_line_views(xview, yview)
    if isnothing(xview) || isnothing(yview)
        error("ContextLinesPanel currently requires viewable vector-like containers.")
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

function _refresh_context_lines!(handle::PanelHandle)
    haskey(handle, :x_obs) || return nothing
    haskey(handle, :y_obs) || return nothing
    now = time()
    if now - handle[:last_update_time] < handle[:update_interval]
        return nothing
    end
    handle[:last_update_time] = now

    xview, yview = _matched_container_views(handle[:x_container], handle[:y_container])
    _validate_context_line_views(xview, yview)
    handle[:x_obs].val = xview
    handle[:y_obs].val = yview
    notify(handle[:x_obs])
    notify(handle[:y_obs])

    if getfield(handle.panel, :autolimits) && haskey(handle, :axis)
        autolimits!(handle[:axis])
    end
    return nothing
end
