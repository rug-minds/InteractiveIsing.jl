@inline _layer_index(layer_idx::Integer) = layer_idx
@inline _layer_index(layer_idx) = layer_idx[]

@inline _with_layer(f, g, layer_idx) =
    inline_layer_dispatch(f, _layer_index(layer_idx), layers(g))

current_layer(g, layer_idx) = _with_layer(identity, g, layer_idx)

_has_layer_selector(g) = length(layers(g)) > 1
_has_layer_selector(g::SingleLayerGraph) = false

@inline _panel_process_algos(algo::Processes.AbstractLoopAlgorithm) = Processes.flat_funcs(algo)
@inline _panel_process_algos(algo) = (algo,)

"""
    _panel_source_algo(g)

Return the algorithm that currently defines the graph's UI-relevant process
shape. Prefer the latest graph process when present, otherwise fall back to the
graph default algorithm.
"""
function _panel_source_algo(g::AbstractSpinGraph)
    graph_processes = processes(g)
    if isempty(graph_processes)
        return g.default_algorithm
    end
    return Processes.getalgo(graph_processes[end])
end

"""
    _panel_has_algo_type(g, target_type)

Return whether the active graph algorithm tree contains a child whose algorithm
type is a subtype of `target_type`.
"""
function _panel_has_algo_type(g::AbstractSpinGraph, target_type::Type)
    for algo in _panel_process_algos(_panel_source_algo(g))
        Processes.algotype(algo) <: target_type && return true
    end
    return false
end

"""
    _panel_supported(g, ::Val{panelkey})

Return whether one optional simulation panel should be mounted for graph `g`.
This may inspect either the current running process or the graph's default
algorithm when no process exists yet.
"""
@inline _panel_supported(g::AbstractSpinGraph, ::Val{panelkey}) where {panelkey} = true
@inline _panel_supported(g::AbstractSpinGraph, ::Val{:interactive_variables}) = !isempty(interactivevars(g))
@inline _panel_supported(g::AbstractSpinGraph, ::Val{:kinetic_time}) =
    !isnothing(_kinetic_time_snapshot(g)) || _panel_has_algo_type(g, KineticMC)

"""
    _mount_panel_if_supported!(handle, key, panel_factory, cell)

Mount one child panel only when the graph/process properties indicate the panel
is relevant for the current simulation.
"""
function _mount_panel_if_supported!(handle::PanelHandle, key::Symbol, panel_factory::Function, cell)
    g = handle[:graph]
    _panel_supported(g, Val(key)) || return nothing
    return panel!(handle, key, panel_factory(), cell)
end

function _register_graph_close!(handle::PanelHandle, g)
    close_graphs = get!(handle.host.data, :close_graphs, IdDict{Any, Bool}())
    haskey(close_graphs, g) && return nothing
    close_graphs[g] = true
    onclose!(handle) do _
        _request_graph_process_close!(g)
    end
    return nothing
end

function _register_process_close!(handle::PanelHandle, process::Processes.AbstractProcess)
    close_processes = get!(handle.host.data, :close_processes, IdDict{Any, Bool}())
    close_processes[process] = true
    onclose!(handle) do _
        _request_process_close!(process)
    end
    return nothing
end

function _register_process_close!(handle::PanelHandle, processes)
    for process in processes
        process isa Processes.AbstractProcess || continue
        _register_process_close!(handle, process)
    end
    return nothing
end

function _set_slider_close!(slider, value)
    try
        set_close_to!(slider, value)
    catch
        slider.value[] = value
    end
    return slider
end

_temperature_value(value::Real) = value
_temperature_value(value::Base.RefValue{<:Real}) = value[]
_temperature_value(value::Processes.InteractiveVar{<:Real}) = value[]
_temperature_value(value::Processes.InteractiveVar{<:Base.RefValue{<:Real}}) = value[][]
_temperature_value(_) = nothing

_interactive_numeric_value(value::Real) = value
_interactive_numeric_value(value::Base.RefValue{<:Real}) = value[]
_interactive_numeric_value(value::Processes.InteractiveVar{<:Real}) = value[]
_interactive_numeric_value(value::Processes.InteractiveVar{<:Base.RefValue{<:Real}}) = value[][]
_interactive_numeric_value(_) = nothing

function _set_interactive_numeric_value!(slot::Base.RefValue{T}, value) where {T<:Real}
    slot[] = convert(T, value)
    return slot[]
end

function _set_interactive_numeric_value!(slot::Processes.InteractiveVar{T}, value) where {T<:Real}
    slot[] = convert(T, value)
    return slot[]
end

function _set_interactive_numeric_value!(slot::Processes.InteractiveVar{<:Base.RefValue{T}}, value) where {T<:Real}
    slot[][] = convert(T, value)
    return slot[][]
end

@inline _set_interactive_numeric_value!(slot, value) = nothing

function _interactive_process_slot(process::Processes.AbstractProcess, spec::InteractiveGraphVarSpec)
    context = Processes.context(process)
    context isa Processes.ProcessContext || return nothing

    target_name = _resolve_interactive_target_key(
        Processes.getregistry(Processes.getalgo(process)),
        spec.target,
    )
    isnothing(target_name) && return nothing
    subcontexts = Processes.get_subcontexts(context)
    hasproperty(subcontexts, target_name) || return nothing

    data = Processes.getdata(getproperty(subcontexts, target_name))
    haskey(data, spec.varname) || return nothing
    return target_name, getproperty(data, spec.varname)
end

function _interactive_prepared_value(g::G, spec::InteractiveGraphVarSpec) where {G<:AbstractSpinGraph}
    func = deepcopy(g.default_algorithm)
    graph_inputs = _mc_model_inits(func, g)
    prepared = Processes.init(Processes.normalize_process_algo(func), graph_inputs...; lifetime = Processes.Indefinite())
    data = _prepared_interactive_var_data(prepared, spec.target, spec.varname)
    isnothing(data) && return nothing
    return _interactive_numeric_value(last(data))
end

function _interactive_graph_var_value(g::G, spec::InteractiveGraphVarSpec) where {G<:AbstractSpinGraph}
    for process in reverse(processes(g))
        slot = _interactive_process_slot(process, spec)
        isnothing(slot) && continue
        value = _interactive_numeric_value(last(slot))
        isnothing(value) || return value
    end

    if !isnothing(spec.value)
        return spec.value
    end

    value = _interactive_prepared_value(g, spec)
    isnothing(value) && return nothing
    _set_interactive_graph_var_value!(g, spec.target, spec.varname, value)
    return value
end

function _set_graph_interactive_var!(g::G, spec::InteractiveGraphVarSpec, value) where {G<:AbstractSpinGraph}
    _set_interactive_graph_var_value!(g, spec.target, spec.varname, value)

    for process in processes(g)
        slot = _interactive_process_slot(process, spec)
        isnothing(slot) && continue
        target_name, current = slot
        if !isnothing(_set_interactive_numeric_value!(current, value))
            continue
        elseif !Processes.isrunning(process)
            converted = convert(typeof(current), value)
            update = NamedTuple{(target_name,)}((NamedTuple{(spec.varname,)}((converted,)),))
            Processes.context(process, Processes.merge_into_subcontexts(Processes.context(process), update))
        end
    end

    return value
end

function _interactive_slider_range(spec::InteractiveGraphVarSpec, value::Real)
    range = spec.range
    if range isa AbstractRange
        return range
    elseif range isa Tuple && length(range) == 2
        lo, hi = Float64(first(range)), Float64(last(range))
        step = value isa Integer ? 1 : max((hi - lo) / 200, eps(Float64))
        if value isa Integer
            return round(Int, lo):1:round(Int, hi)
        else
            return lo:step:hi
        end
    elseif !isnothing(range)
        return range
    end

    if value isa Integer
        hi = max(10, 4 * abs(Int(value)))
        return 0:1:hi
    end

    positive_name = spec.varname in (:T, :temp, :stepsize, :max_drift_fraction, :block_size, :group_steps, :max_blocksize, :langevin_steps)
    if positive_name || value >= 0
        hi = max(1.0, 4 * abs(Float64(value)))
        return 0.0:max(hi / 200, 0.001):hi
    end

    span = max(1.0, 2 * abs(Float64(value)))
    return (-span):max((2 * span) / 200, 0.001):span
end

"""
    _interactive_range_limits(range)

Return the lower and upper numeric bounds of one interactive slider range.
"""
function _interactive_range_limits(range::AbstractRange)
    lo = Float64(first(range))
    hi = Float64(last(range))
    return min(lo, hi), max(lo, hi)
end

"""
    _interactive_default_delta(range, value)

Choose a default `+/-` increment for one interactive variable.
"""
function _interactive_default_delta(range::AbstractRange, value::T) where {T<:Real}
    if value isa Integer
        return one(T)
    end

    step_value = try
        Float64(step(range))
    catch
        0.0
    end
    if isfinite(step_value) && step_value > 0
        return T(step_value)
    end

    lo, hi = _interactive_range_limits(range)
    return T(max((hi - lo) / 100, 0.001))
end

"""
    _parse_interactive_delta(text, current, fallback)

Parse one delta textbox value, falling back to the previous delta when the
textbox input is empty or invalid.
"""
function _parse_interactive_delta(text, current::T, fallback::T) where {T<:Real}
    isnothing(text) && return fallback
    stripped = strip(text)
    isempty(stripped) && return fallback

    parsed = current isa Integer ? tryparse(Int, stripped) : tryparse(Float64, stripped)
    isnothing(parsed) && return fallback

    delta = current isa Integer ? abs(parsed) : abs(T(parsed))
    return delta > zero(T) ? delta : fallback
end

"""
    _interactive_quantize_value(value, range)

Snap one candidate value onto the slider range grid and clamp it to the valid
range interval.
"""
function _interactive_quantize_value(value::T, range::AbstractRange) where {T<:Real}
    lo, hi = _interactive_range_limits(range)
    clamped = clamp(Float64(value), lo, hi)

    if value isa Integer
        return T(round(Int, clamped))
    end

    step_value = try
        abs(Float64(step(range)))
    catch
        0.0
    end
    if !(isfinite(step_value) && step_value > 0)
        return T(clamped)
    end

    base = Float64(first(range))
    snapped = base + round((clamped - base) / step_value) * step_value
    return T(clamp(snapped, lo, hi))
end

"""
    _interactive_commit_delta!(textbox, delta, current)

Commit the current textbox contents immediately, even when the user clicks a
button without pressing Enter first.
"""
function _interactive_commit_delta!(textbox, delta, current::T) where {T<:Real}
    text = textbox.displayed_string[]
    delta_type = typeof(delta[])
    current_value = convert(delta_type, current)
    delta[] = _parse_interactive_delta(text, current_value, delta[])
    textbox.displayed_string[] = string(delta[])
    return delta[]
end

"""
    _interactive_adjusted_value(value, delta, direction, range)

Apply one signed delta step and snap the result to the slider range.
"""
function _interactive_adjusted_value(value::T, delta::T, direction::Int, range::AbstractRange) where {T<:Real}
    raw = value + direction * delta
    return _interactive_quantize_value(raw, range)
end

function _temperature_vars(sc)
    data = Processes.getdata(sc)
    pairs = Pair{Symbol, Any}[]
    for name in (:temp, :T)
        haskey(data, name) || continue
        value = getproperty(data, name)
        isnothing(_temperature_value(value)) || push!(pairs, name => value)
    end
    return pairs
end

function _process_context_temperature(process::Processes.AbstractProcess)
    context = Processes.context(process)
    context isa Processes.ProcessContext || return nothing

    subcontexts = Processes.get_subcontexts(context)
    for subcontext_name in reverse(propertynames(subcontexts))
        subcontext_name === :globals && continue
        subcontext_name === :_injector && continue
        subcontext_name === :_exchange && continue
        vars = _temperature_vars(getproperty(subcontexts, subcontext_name))
        isempty(vars) && continue
        return _temperature_value(last(vars).second)
    end
    return nothing
end

function _process_context_temperature(g::AbstractSpinGraph)
    for process in reverse(processes(g))
        value = _process_context_temperature(process)
        isnothing(value) || return value
    end
    return nothing
end

function _set_process_context_temperature!(process::Processes.AbstractProcess, value)
    context = Processes.context(process)
    context isa Processes.ProcessContext || return nothing

    subcontexts = Processes.get_subcontexts(context)
    for subcontext_name in propertynames(subcontexts)
        subcontext_name === :globals && continue
        subcontext_name === :_injector && continue
        subcontext_name === :_exchange && continue

        subcontext = getproperty(subcontexts, subcontext_name)
        for (varname, current) in _temperature_vars(subcontext)
            if current isa Base.RefValue || current isa Processes.InteractiveVar
                current[] = convert(typeof(current[]), value)
            elseif Processes.isinteractive(process)
                Processes.interact!(process, varname => value)
            elseif !Processes.isrunning(process)
                converted = convert(typeof(current), value)
                update = NamedTuple{(subcontext_name,)}((NamedTuple{(varname,)}((converted,)),))
                Processes.context(process, Processes.merge_into_subcontexts(context, update))
            end
        end
    end
    return nothing
end

function _set_temperature!(g, value)
    temp!(g, value)
    for process in processes(g)
        _set_process_context_temperature!(process, value)
    end
    return value
end

function _poll_temperature!(g, last_graph_temp, last_context_temp)
    graph_temp = temp(g)
    context_temp = _process_context_temperature(g)

    if graph_temp != last_graph_temp[]
        last_graph_temp[] = graph_temp
        last_context_temp[] = context_temp
        return graph_temp
    elseif !isnothing(context_temp) && context_temp != last_context_temp[]
        last_context_temp[] = context_temp
        temp!(g, context_temp)
        last_graph_temp[] = temp(g)
        return context_temp
    end

    return graph_temp
end

_kinetic_number(value::Real) = Float64(value)
_kinetic_number(value::Base.RefValue{<:Real}) = Float64(value[])
_kinetic_number(_) = nothing

function _kinetic_totalrate(data)
    if haskey(data, :events)
        return _kinetic_number(getproperty(data, :events).totalrate)
    elseif haskey(data, :rates)
        return _kinetic_number(getproperty(data, :rates).totalrate)
    elseif haskey(data, :totalrate)
        return _kinetic_number(getproperty(data, :totalrate))
    end
    return nothing
end

function _kinetic_lastdt(data)
    if haskey(data, :lastdt)
        return _kinetic_number(getproperty(data, :lastdt))
    elseif haskey(data, :dt)
        return _kinetic_number(getproperty(data, :dt))
    end
    return nothing
end

function _kinetic_time_snapshot(sc)
    data = Processes.getdata(sc)
    has_time = haskey(data, :time) || haskey(data, :kmc_time)
    has_rate_table = haskey(data, :events) || haskey(data, :rates)
    (has_time || has_rate_table) || return nothing

    kmc_time = haskey(data, :time) ? _kinetic_number(getproperty(data, :time)) :
        _kinetic_number(getproperty(data, :kmc_time))
    isnothing(kmc_time) && return nothing

    dt = _kinetic_lastdt(data)
    totalrate = _kinetic_totalrate(data)
    return (; time = kmc_time, dt, totalrate)
end

function _kinetic_time_snapshot(process::Processes.AbstractProcess)
    context = try
        Processes.getcontext(process)
    catch
        Processes.context(process)
    end
    context isa Processes.ProcessContext || return nothing

    subcontexts = Processes.get_subcontexts(context)
    for subcontext_name in reverse(propertynames(subcontexts))
        subcontext_name === :globals && continue
        subcontext_name === :_injector && continue
        snapshot = _kinetic_time_snapshot(getproperty(subcontexts, subcontext_name))
        isnothing(snapshot) || return snapshot
    end
    return nothing
end

function _kinetic_time_snapshot(g::AbstractSpinGraph)
    for process in reverse(processes(g))
        snapshot = _kinetic_time_snapshot(process)
        isnothing(snapshot) || return snapshot
    end
    return nothing
end

function _format_kinetic_value(value)
    isnothing(value) && return "-"
    absvalue = abs(value)
    if absvalue != 0 && (absvalue < 0.001 || absvalue >= 10000)
        return string(round(value; sigdigits = 4))
    end
    return string(round(value; digits = 4))
end

function _kinetic_time_label(g)
    snapshot = _kinetic_time_snapshot(g)
    isnothing(snapshot) && return "KMC time\n-"

    lines = ["KMC time", _format_kinetic_value(snapshot.time)]
    isnothing(snapshot.dt) || push!(lines, "dt " * _format_kinetic_value(snapshot.dt))
    isnothing(snapshot.totalrate) || push!(lines, "rate " * _format_kinetic_value(snapshot.totalrate))
    return join(lines, "\n")
end

function _total_ticks(g)
    total = 0
    for process in processes(g)
        total += Int(Processes.getticks(process))
    end
    return total
end

function _pause_graph_processes!(g)
    for process in processes(g)
        if Processes.isrunning(process)
            Processes.pause(process)
        end
    end
    return nothing
end

function _resume_graph_processes!(g)
    for process in processes(g)
        if Processes.ispaused(process)
            run(process)
        end
    end
    return nothing
end

function _graph_paused(g)
    graph_processes = processes(g)
    isempty(graph_processes) && return false
    any(Processes.isrunning, graph_processes) && return false
    return any(Processes.ispaused, graph_processes)
end

function _request_graph_process_close!(g)
    graph_processes = collect(processes(g))
    isempty(graph_processes) && return nothing

    for process in graph_processes
        _request_process_close!(process)
    end

    empty!(processes(g))
    _reap_graph_processes_later!(graph_processes)

    return nothing
end

function _request_process_close!(process::Processes.AbstractProcess)
    try
        if applicable(Processes.shouldrun, process, false)
            Processes.shouldrun(process, false)
        elseif applicable(close, process)
            @async close(process)
        end
    catch err
        @warn "Could not request process stop" process_type = typeof(process) exception = (err, catch_backtrace())
    end
    return nothing
end

function _reap_graph_processes_later!(graph_processes)
    @async begin
        while true
            any_running = false
            for process in graph_processes
                try
                    any_running |= Processes.isrunning(process)
                catch
                end
            end
            any_running || break
            sleep(0.05)
        end

        for process in graph_processes
            try
                process.task = nothing
                process.loopidx = 1
            catch err
                @warn "Could not finalize stopped graph process" process exception = (err, catch_backtrace())
            end
        end
    end
    return nothing
end

function _magnetization(g, layer_idx)
    return _with_layer(g, layer_idx) do layer
        layer_graph = graph(layer)
        if layer_graph isa AbstractVectorSpinGraph
            return norm(sum(state(layer))) / max(1, length(state(layer)))
        end
        return sum(state(layer))
    end
end

function _layer_colorrange(layer)
    if graph(layer) isa AbstractVectorSpinGraph
        return _vector_spin_display_colorrange(graph(layer))
    end

    range = stateset(layer)
    if isfinite(range[1]) && isfinite(range[end]) && range[1] < range[end]
        return (range[1], range[end])
    end

    T = eltype(state(layer))
    return (T(-1), T(1))
end

function _bind_layer_colorrange!(plot, state_obs, layer)
    plot.colorrange[] = _layer_colorrange(layer)
    return plot
end

_is_axis3_like(axis) = hasproperty(axis, :azimuth) && hasproperty(axis, :elevation)

function _axis3_state(axis)
    _is_axis3_like(axis) || return nothing

    state = Dict{Symbol, Any}()
    for key in (:azimuth, :elevation, :perspectiveness, :zoom_mult, :targetlimits, :limits)
        hasproperty(axis, key) || continue
        attr = getproperty(axis, key)
        if attr isa Observable
            state[key] = deepcopy(attr[])
        end
    end
    return state
end

function _restore_axis3_state!(axis, state)
    isnothing(state) && return axis
    _is_axis3_like(axis) || return axis

    for (key, value) in state
        hasproperty(axis, key) || continue
        attr = getproperty(axis, key)
        attr isa Observable || continue
        try
            attr[] = value
        catch
        end
    end
    return axis
end

function _remember_axis3_state!(handle, key::Symbol, axis)
    state = _axis3_state(axis)
    isnothing(state) || (handle[key] = state)
    return state
end

function _delete_makie_object!(handle, object)
    host = handle.host
    (host.closed || host.closing) && return nothing
    try
        delete!(object)
    catch
    end
    return nothing
end

function _old_linear_layer_coordinates(vals_size)
    sx, sy, sz = vals_size
    allidxs = [1:(sx * sy * sz);]
    xs = [(idx - 1) % sx + 1 for idx in allidxs]
    ys = [floor(Int, (idx - 1) / sx) % sy + 1 for idx in allidxs]
    zs = [floor(Int, (idx - 1) / (sx * sy)) + 1 for idx in allidxs]
    return xs, ys, zs
end

function _coordinates_3d!(handle, vals_size)
    cache = get!(handle.data, :coordinates_3d) do
        Dict{Any, Any}()
    end
    return get!(cache, vals_size) do
        _old_linear_layer_coordinates(vals_size)
    end
end

"""
    _coordinates_3d!(handle, layer)

Return cached 3D display coordinates for a layer using its topology geometry.
"""
function _coordinates_3d!(handle, layer::AbstractIsingLayer)
    return _coordinates_3d!(handle, topology(layer), size(layer))
end

"""
    _coordinates_3d!(handle, top, vals_size)

Return 3D display coordinates for a three-dimensional topology.
"""
function _coordinates_3d!(handle, top::AbstractLayerTopology{U,3}, vals_size::NTuple{3,<:Integer}) where {U}
    return _world_layer_coordinates_3d(top, vals_size)
end

"""
    _world_layer_coordinates_3d(top, vals_size)

Return world-space `x`, `y`, and `z` coordinate vectors for a 3D topology.
"""
function _world_layer_coordinates_3d(top::T, vals_size::NTuple{3,<:Integer}) where {U,T<:AbstractLayerTopology{U,3}}
    len = prod(vals_size)
    xs = Vector{Float32}(undef, len)
    ys = Vector{Float32}(undef, len)
    zs = Vector{Float32}(undef, len)
    linear = LinearIndices(vals_size)

    for ci in CartesianIndices(vals_size)
        wc = woorldcoordinate(top, Coordinate(top, ci; check = false))
        idx = linear[ci]
        xs[idx] = Float32(wc[1])
        ys[idx] = Float32(wc[2])
        zs[idx] = Float32(wc[3])
    end
    return xs, ys, zs
end

struct _VectorSpinDisplayArray{A,T,N} <: AbstractArray{T,N}
    data::A
end

Base.size(display::_VectorSpinDisplayArray) = size(display.data)
Base.IndexStyle(::Type{<:_VectorSpinDisplayArray}) = IndexCartesian()

@inline function Base.getindex(display::_VectorSpinDisplayArray, idxs...)
    spin = @inbounds display.data[idxs...]
    return _vector_spin_display_value(spin)
end

function hot_observable_zero(::Type{T}) where {A,S,N,T<:_VectorSpinDisplayArray{A,S,N}}
    data = Array{eltype(A),N}(undef, ntuple(_ -> 0, Val(N)))
    replacement = _VectorSpinDisplayArray{typeof(data),S,N}(data)
    replacement isa T && return replacement
    throw(ArgumentError("Cannot build zero-sized replacement for hot observable value type $T."))
end

"""
    _vector_spin_display_value(spin)

Return the scalar color value used for vector-spin layer displays.

Two or more components are displayed by in-plane angle `atan(s_y, s_x)`.
Single-component vector spins fall back to their only component.
"""
@inline function _vector_spin_display_value(spin)
    length(spin) == 1 && return Float32(spin[1])
    return Float32(atan(spin[2], spin[1]))
end

@inline function _vector_spin_display_colorrange(g::AbstractVectorSpinGraph)
    spin_dimension(g) == 1 && return (-1f0, 1f0)
    return (Float32(-π), Float32(π))
end

@inline _is_vector_spin_2d_layer(layer::AbstractIsingLayer{T,2}) where {T} =
    graph(layer) isa AbstractVectorSpinGraph && spin_dimension(graph(layer)) >= 2
@inline _is_vector_spin_2d_layer(layer) = false

@inline _is_vector_spin_3d_layer(layer::AbstractIsingLayer{T,3}) where {T} =
    graph(layer) isa AbstractVectorSpinGraph && spin_dimension(graph(layer)) >= 3
@inline _is_vector_spin_3d_layer(layer) = false

@inline function _vector_arrow_stride(layer)
    max_side = maximum(size(layer))
    return max(1, cld(Int(max_side), 16))
end

"""
    _vector_arrow_strides_3d(layer)

Return the visual sampling strides for 3D vector-spin arrows.

The simulation still runs on the full lattice. Only the horizontal axes are
sampled for dense volumes; the vertical axis is always shown at full
resolution so shallow 3D systems do not collapse to a few displayed planes.
"""
@inline function _vector_arrow_strides_3d(layer::L) where {L<:AbstractIsingLayer}
    sx, sy, _ = size(layer)
    horizontal_stride = max(1, cld(Int(max(sx, sy)), 10))
    return horizontal_stride, horizontal_stride, 1
end

"""
    _vector_spin_glyphs_2d(layer)

Return downsampled arrow positions and direction vectors for a visible
vector-spin layer display.
"""
function _vector_spin_glyphs_2d(layer::AbstractIsingLayer{T,2}) where {T}
    layer_graph = graph(layer)
    layer_graph isa AbstractVectorSpinGraph || return nothing
    spin_dimension(layer_graph) >= 2 || return nothing

    spin_state = state(layer)
    stride = _vector_arrow_stride(layer)
    arrow_scale = Float32(0.62 * stride)
    positions = Point2f[]
    directions = Vec2f[]

    # Makie image coordinates are cell-centered at integer positions here.
    @inbounds for j in 1:stride:size(layer, 2), i in 1:stride:size(layer, 1)
        spin = spin_state[i, j]
        push!(positions, Point2f(Float32(i), Float32(j)))
        push!(directions, Vec2f(arrow_scale * Float32(spin[1]), arrow_scale * Float32(spin[2])))
    end
    return positions, directions
end
_vector_spin_glyphs_2d(layer) = nothing

function _vector_arrow_key(prefix::Symbol, suffix::Symbol)
    return Symbol(prefix, :_, suffix)
end

function _draw_vector_spin_arrows_2d!(handle, ax, layer; prefix::Symbol = :vector_arrow)
    geometry = _vector_spin_glyphs_2d(layer)
    isnothing(geometry) && return nothing

    layer_key = _vector_arrow_key(prefix, :layer)
    positions_key = _vector_arrow_key(prefix, :positions)
    directions_key = _vector_arrow_key(prefix, :directions)
    underlay_key = _vector_arrow_key(prefix, :underlay)
    plot_key = _vector_arrow_key(prefix, :plot)

    handle[layer_key] = layer
    handle[positions_key] = Observable(first(geometry))
    handle[directions_key] = Observable(last(geometry))
    handle[underlay_key] = arrows2d!(
        ax,
        handle[positions_key],
        handle[directions_key];
        align = :center,
        color = (:white, 0.75),
        shaftwidth = 5,
        tipwidth = 16,
        tiplength = 11,
        minshaftlength = 0,
    )
    handle[plot_key] = arrows2d!(
        ax,
        handle[positions_key],
        handle[directions_key];
        align = :center,
        color = (:black, 0.95),
        shaftwidth = 2.6,
        tipwidth = 11,
        tiplength = 8,
        minshaftlength = 0,
    )
    return handle[plot_key]
end

function _draw_vector_spin_layer_2d!(
    handle,
    cell,
    layer;
    axis_key::Symbol = :axis,
    prefix::Symbol = :vector_arrow,
    yflip_default::Bool = true,
)
    ax = handle[axis_key] = Axis(cell, xrectzoom = false, yrectzoom = false, aspect = DataAspect(), tellheight = true)
    ax.yreversed = @load_preference("makie_y_flip", default = yflip_default)
    _draw_vector_spin_arrows_2d!(handle, ax, layer; prefix)
    xlims!(ax, 0.5, size(layer, 1) + 0.5)
    ylims!(ax, 0.5, size(layer, 2) + 0.5)
    return handle
end

"""
    _vector_spin_glyphs_3d(handle, layer)

Return downsampled 3D arrow positions, direction vectors, and magnitudes for a
vector-spin layer display.
"""
function _vector_spin_glyphs_3d(handle, layer::AbstractIsingLayer{T,3}) where {T}
    layer_graph = graph(layer)
    layer_graph isa AbstractVectorSpinGraph || return nothing
    spin_dimension(layer_graph) >= 3 || return nothing

    spin_state = state(layer)
    xstride, ystride, zstride = _vector_arrow_strides_3d(layer)
    xs, ys, zs = _coordinates_3d!(handle, layer)
    linear = LinearIndices(size(layer))
    positions = Point3f[]
    directions = Vec3f[]
    magnitudes = Float32[]

    # Direction vectors keep their state magnitude; color displays the same magnitude.
    @inbounds for k in 1:zstride:size(layer, 3), j in 1:ystride:size(layer, 2), i in 1:xstride:size(layer, 1)
        idx = linear[i, j, k]
        spin = spin_state[i, j, k]
        spin_direction = Vec3f(Float32(spin[1]), Float32(spin[2]), Float32(spin[3]))
        spin_norm = norm(spin_direction)
        spin_norm <= eps(Float32) && continue

        push!(positions, Point3f(xs[idx], ys[idx], zs[idx]))
        push!(directions, spin_direction)
        push!(magnitudes, Float32(spin_norm))
    end
    return (;positions, directions, magnitudes)
end
_vector_spin_glyphs_3d(handle, layer) = nothing

"""
    _vector_spin_magnitude_slice_3d(layer)

Return a central z-slice of vector magnitudes for the side heatmap.
"""
function _vector_spin_magnitude_slice_3d(layer::AbstractIsingLayer{T,3}) where {T}
    spin_state = state(layer)
    zidx = cld(size(layer, 3), 2)
    values = Matrix{Float32}(undef, size(layer, 1), size(layer, 2))

    @inbounds for j in 1:size(layer, 2), i in 1:size(layer, 1)
        values[i, j] = Float32(norm(spin_state[i, j, zidx]))
    end
    return values
end

@inline function _vector_spin_magnitude_colorrange(values)
    isempty(values) && return (0f0, 1f0)
    hi = maximum(values)
    return (0f0, max(Float32(hi), eps(Float32)))
end

function _draw_vector_spin_arrows_3d!(handle, ax, layer; prefix::Symbol = :vector_arrow)
    geometry = _vector_spin_glyphs_3d(handle, layer)
    isnothing(geometry) && return nothing

    layer_key = _vector_arrow_key(prefix, :layer)
    positions_key = _vector_arrow_key(prefix, :positions)
    directions_key = _vector_arrow_key(prefix, :directions)
    magnitudes_key = _vector_arrow_key(prefix, :magnitudes)
    plot_key = _vector_arrow_key(prefix, :plot)

    handle[layer_key] = layer
    handle[positions_key] = Observable(geometry.positions)
    handle[directions_key] = Observable(geometry.directions)
    handle[magnitudes_key] = Observable(geometry.magnitudes)
    handle[plot_key] = arrows3d!(
        ax,
        handle[positions_key],
        handle[directions_key];
        color = handle[magnitudes_key],
        colormap = :viridis,
        colorrange = _vector_spin_magnitude_colorrange(geometry.magnitudes),
        align = :center,
        normalize = false,
        lengthscale = 1.5f0,
        markerscale = 1.0f0,
        minshaftlength = 0.0,
        shaftradius = 0.025,
        tipradius = 0.075,
        tiplength = 0.22,
    )
    return handle[plot_key]
end

function _draw_vector_spin_magnitude_heatmap_3d!(handle, cell, layer; prefix::Symbol = :vector_arrow)
    values = _vector_spin_magnitude_slice_3d(layer)
    obs_key = _vector_arrow_key(prefix, :magnitude_heatmap_obs)
    plot_key = _vector_arrow_key(prefix, :magnitude_heatmap_plot)
    axis_key = _vector_arrow_key(prefix, :magnitude_axis)

    ax = handle[axis_key] = Axis(cell, aspect = DataAspect(), tellheight = true, tellwidth = true)
    ax.yreversed = @load_preference("makie_y_flip", default = false)
    handle[obs_key] = Observable(values)
    plot = handle[plot_key] = image!(
        ax,
        handle[obs_key];
        colormap = :viridis,
        colorrange = _vector_spin_magnitude_colorrange(values),
        fxaa = false,
        interpolate = false,
    )
    hidedecorations!(ax; grid = false)
    hidespines!(ax)
    return plot
end

function _draw_vector_spin_layer_3d!(
    handle,
    cell,
    layer;
    axis_key::Symbol = :axis,
    prefix::Symbol = :vector_arrow,
)
    grid = GridLayout(cell)
    _draw_vector_spin_magnitude_heatmap_3d!(handle, grid[1, 1], layer; prefix)
    ax = handle[axis_key] = Axis3(grid[1, 2], tellheight = true)
    colsize!(grid, 1, Relative(0.24))
    colsize!(grid, 2, Auto(false))
    _restore_axis3_state!(ax, get(handle.data, Symbol(String(axis_key), "3_state"), nothing))
    _draw_vector_spin_arrows_3d!(handle, ax, layer; prefix)
    return handle
end

function _refresh_vector_spin_arrows_2d!(handle; prefix::Symbol = :vector_arrow)
    positions_key = _vector_arrow_key(prefix, :positions)
    haskey(handle, positions_key) || return nothing
    layer = handle[_vector_arrow_key(prefix, :layer)]
    geometry = _vector_spin_glyphs_2d(layer)
    isnothing(geometry) && return nothing
    positions, directions = geometry
    handle[positions_key][] = positions
    handle[_vector_arrow_key(prefix, :directions)][] = directions
    return nothing
end

function _refresh_vector_spin_arrows_3d!(handle; prefix::Symbol = :vector_arrow)
    positions_key = _vector_arrow_key(prefix, :positions)
    haskey(handle, positions_key) || return nothing
    layer = handle[_vector_arrow_key(prefix, :layer)]
    _is_vector_spin_3d_layer(layer) || return nothing
    geometry = _vector_spin_glyphs_3d(handle, layer)
    isnothing(geometry) && return nothing
    handle[positions_key][] = geometry.positions
    handle[_vector_arrow_key(prefix, :directions)][] = geometry.directions
    handle[_vector_arrow_key(prefix, :magnitudes)][] = geometry.magnitudes
    handle[_vector_arrow_key(prefix, :plot)].colorrange[] = _vector_spin_magnitude_colorrange(geometry.magnitudes)

    heatmap_obs_key = _vector_arrow_key(prefix, :magnitude_heatmap_obs)
    if haskey(handle, heatmap_obs_key)
        values = _vector_spin_magnitude_slice_3d(layer)
        handle[heatmap_obs_key][] = values
        handle[_vector_arrow_key(prefix, :magnitude_heatmap_plot)].colorrange[] = _vector_spin_magnitude_colorrange(values)
    end
    return nothing
end

function _refresh_vector_spin_arrows!(handle; prefix::Symbol = :vector_arrow)
    _refresh_vector_spin_arrows_2d!(handle; prefix)
    _refresh_vector_spin_arrows_3d!(handle; prefix)
    return nothing
end

_layer_state_view(layer) = graph(layer) isa AbstractVectorSpinGraph ?
    _VectorSpinDisplayArray{typeof(state(layer)),Float32,ndims(state(layer))}(state(layer)) :
    view(state(layer), ntuple(_ -> (:), ndims(state(layer)))...)
_layer_state_vector_view(layer) = vec(_layer_state_view(layer))
_layer_state_float_vector(layer) = Float64.(vec(_layer_state_view(layer)))

function _cast_layer_state_vector(layer)
    if graph(layer) isa AbstractVectorSpinGraph
        return Float64.(vec(_layer_state_view(layer)))
    end

    layer_state = state(layer)
    unsafe_vector = unsafe_wrap(Vector{eltype(layer_state)}, pointer(layer_state), length(layer_state))
    return CastVec(Float64, unsafe_vector)
end
