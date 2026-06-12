@inline _layer_index(layer_idx::Integer) = layer_idx
@inline _layer_index(layer_idx) = layer_idx[]

@inline _with_layer(f, g, layer_idx) =
    inline_layer_dispatch(f, _layer_index(layer_idx), layers(g))

current_layer(g, layer_idx) = _with_layer(identity, g, layer_idx)

_has_layer_selector(g) = length(layers(g)) > 1
_has_layer_selector(g::SingleLayerGraph) = false

@inline _panel_process_algos(algo::StatefulAlgorithms.AbstractLoopAlgorithm) = StatefulAlgorithms.flat_funcs(algo)
@inline _panel_process_algos(algo) = (algo,)

"""
    _panel_source_algo(g)

Return the algorithm that currently defines the graph's UI-relevant process
shape. Prefer the latest graph process when present, otherwise fall back to the
graph default algorithm.
"""
function _panel_source_algo(g::IsingGraph)
    graph_processes = processes(g)
    if isempty(graph_processes)
        return g.default_algorithm
    end
    return StatefulAlgorithms.getalgo(graph_processes[end])
end

"""
    _panel_has_algo_type(g, target_type)

Return whether the active graph algorithm tree contains a child whose algorithm
type is a subtype of `target_type`.
"""
function _panel_has_algo_type(g::IsingGraph, target_type::Type)
    for algo in _panel_process_algos(_panel_source_algo(g))
        StatefulAlgorithms.algotype(algo) <: target_type && return true
    end
    return false
end

"""
    _panel_supported(g, ::Val{panelkey})

Return whether one optional simulation panel should be mounted for graph `g`.
This may inspect either the current running process or the graph's default
algorithm when no process exists yet.
"""
@inline _panel_supported(g::IsingGraph, ::Val{panelkey}) where {panelkey} = true
@inline _panel_supported(g::IsingGraph, ::Val{:interactive_variables}) = !isempty(interactivevars(g))
@inline _panel_supported(g::IsingGraph, ::Val{:kinetic_time}) =
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

function _register_process_close!(handle::PanelHandle, process::StatefulAlgorithms.AbstractProcess)
    close_processes = get!(handle.host.data, :close_processes, IdDict{Any, Bool}())
    close_processes[process] = true
    onclose!(handle) do _
        _request_process_close!(process)
    end
    return nothing
end

function _register_process_close!(handle::PanelHandle, processes)
    for process in processes
        process isa StatefulAlgorithms.AbstractProcess || continue
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
_temperature_value(value::StatefulAlgorithms.InteractiveVar{<:Real}) = value[]
_temperature_value(value::StatefulAlgorithms.InteractiveVar{<:Base.RefValue{<:Real}}) = value[][]
_temperature_value(_) = nothing

"""
    _temperature_display_scale(g)

Return the physical scale used only for displaying graph temperature. An
explicit temperature scale wins. Energy-like scales are interpreted as `k_B T`
and converted to kelvin for display only.
"""
function _temperature_display_scale(g::G) where {G<:IsingGraph}
    scales = physicalscales(g)
    temperature_scale = scales.temperature[]
    isnothing(temperature_scale) || return _temperature_kelvin_display_scale(temperature_scale)
    return _temperature_kelvin_display_scale(scales.energy[])
end

"""
    _temperature_kelvin_display_scale(scale)

Return a display scale that shows energy-like `k_B T` units as kelvin while
leaving non-convertible scales unchanged.
"""
function _temperature_kelvin_display_scale(scale::Unitful.AbstractQuantity)
    try
        return Unitful.uconvert(u"K", scale / Unitful.k)
    catch
        nothing
    end

    try
        return Unitful.uconvert(u"K", scale)
    catch
        return scale
    end
end

"""
    _temperature_kelvin_display_scale(scale::Unitful.Units)

Promote a bare Unitful unit object to a unit quantity before applying the
temperature display conversion.
"""
function _temperature_kelvin_display_scale(scale::U) where {U<:Unitful.Units}
    return _temperature_kelvin_display_scale(1 * scale)
end

_temperature_kelvin_display_scale(scale) = scale

"""
    _temperature_number_text(value)

Format one real temperature coefficient compactly for UI labels.
"""
function _temperature_number_text(value::T) where {T<:Real}
    numeric = Float64(value)
    isfinite(numeric) || return string(value)
    return string(round(numeric, sigdigits = 4))
end

"""
    _temperature_display_text(value, scale)

Format an internal temperature against one Unitful display scale.
"""
function _temperature_display_text(value::T, scale::Unitful.AbstractQuantity) where {T<:Real}
    quantity = value * scale
    return "$(_temperature_number_text(Unitful.ustrip(quantity))) $(Unitful.unit(quantity))"
end

"""
    _temperature_display_text(value, scale)

Format an internal temperature against a non-Unitful scale or without a scale.
"""
function _temperature_display_text(value::T, scale) where {T<:Real}
    isnothing(scale) && return _temperature_number_text(value)
    scaled = value * scale
    scaled isa Real || return string(scaled)
    return _temperature_number_text(scaled)
end

"""
    _temperature_label(g, value)

Format the temperature label shown by the interface without changing the
internal slider/process value.
"""
function _temperature_label(g::G, value::T) where {G<:IsingGraph,T<:Real}
    return "T: $(_temperature_display_text(value, _temperature_display_scale(g)))"
end

_interactive_numeric_value(value::Real) = value
_interactive_numeric_value(value::Base.RefValue{<:Real}) = value[]
_interactive_numeric_value(value::StatefulAlgorithms.InteractiveVar{<:Real}) = value[]
_interactive_numeric_value(value::StatefulAlgorithms.InteractiveVar{<:Base.RefValue{<:Real}}) = value[][]
_interactive_numeric_value(_) = nothing

function _set_interactive_numeric_value!(slot::Base.RefValue{T}, value) where {T<:Real}
    slot[] = convert(T, value)
    return slot[]
end

function _set_interactive_numeric_value!(slot::StatefulAlgorithms.InteractiveVar{T}, value) where {T<:Real}
    slot[] = convert(T, value)
    return slot[]
end

function _set_interactive_numeric_value!(slot::StatefulAlgorithms.InteractiveVar{<:Base.RefValue{T}}, value) where {T<:Real}
    slot[][] = convert(T, value)
    return slot[][]
end

@inline _set_interactive_numeric_value!(slot, value) = nothing

function _interactive_process_slot(process::StatefulAlgorithms.AbstractProcess, spec::InteractiveGraphVarSpec)
    context = StatefulAlgorithms.context(process)
    context isa StatefulAlgorithms.ProcessContext || return nothing

    target_name = _resolve_interactive_target_key(
        StatefulAlgorithms.getregistry(StatefulAlgorithms.getalgo(process)),
        spec.target,
    )
    isnothing(target_name) && return nothing
    subcontexts = StatefulAlgorithms.get_subcontexts(context)
    hasproperty(subcontexts, target_name) || return nothing

    data = StatefulAlgorithms.getdata(getproperty(subcontexts, target_name))
    haskey(data, spec.varname) || return nothing
    return target_name, getproperty(data, spec.varname)
end

function _interactive_prepared_value(g::IsingGraph, spec::InteractiveGraphVarSpec)
    func = deepcopy(g.default_algorithm)
    graph_inputs = _mc_model_inits(func, g)
    prepared = StatefulAlgorithms.init(StatefulAlgorithms.normalize_process_algo(func), graph_inputs...; lifetime = StatefulAlgorithms.Indefinite())
    data = _prepared_interactive_var_data(prepared, spec.target, spec.varname)
    isnothing(data) && return nothing
    return _interactive_numeric_value(last(data))
end

function _interactive_graph_var_value(g::IsingGraph, spec::InteractiveGraphVarSpec)
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

function _set_graph_interactive_var!(g::IsingGraph, spec::InteractiveGraphVarSpec, value)
    _set_interactive_graph_var_value!(g, spec.target, spec.varname, value)

    for process in processes(g)
        slot = _interactive_process_slot(process, spec)
        isnothing(slot) && continue
        target_name, current = slot
        if !isnothing(_set_interactive_numeric_value!(current, value))
            continue
        elseif !StatefulAlgorithms.isrunning(process)
            converted = convert(typeof(current), value)
            update = NamedTuple{(target_name,)}((NamedTuple{(spec.varname,)}((converted,)),))
            StatefulAlgorithms.context(process, StatefulAlgorithms.merge_into_subcontexts(StatefulAlgorithms.context(process), update))
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
    data = StatefulAlgorithms.getdata(sc)
    pairs = Pair{Symbol, Any}[]
    for name in (:temp, :T)
        haskey(data, name) || continue
        value = getproperty(data, name)
        isnothing(_temperature_value(value)) || push!(pairs, name => value)
    end
    return pairs
end

function _process_context_temperature(process::StatefulAlgorithms.AbstractProcess)
    context = StatefulAlgorithms.context(process)
    context isa StatefulAlgorithms.ProcessContext || return nothing

    subcontexts = StatefulAlgorithms.get_subcontexts(context)
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

function _process_context_temperature(g::IsingGraph)
    for process in reverse(processes(g))
        value = _process_context_temperature(process)
        isnothing(value) || return value
    end
    return nothing
end

function _set_process_context_temperature!(process::StatefulAlgorithms.AbstractProcess, value)
    context = StatefulAlgorithms.context(process)
    context isa StatefulAlgorithms.ProcessContext || return nothing

    subcontexts = StatefulAlgorithms.get_subcontexts(context)
    for subcontext_name in propertynames(subcontexts)
        subcontext_name === :globals && continue
        subcontext_name === :_injector && continue
        subcontext_name === :_exchange && continue

        subcontext = getproperty(subcontexts, subcontext_name)
        for (varname, current) in _temperature_vars(subcontext)
            if current isa Base.RefValue || current isa StatefulAlgorithms.InteractiveVar
                current[] = convert(typeof(current[]), value)
            elseif StatefulAlgorithms.isinteractive(process)
                StatefulAlgorithms.interact!(process, varname => value)
            elseif !StatefulAlgorithms.isrunning(process)
                converted = convert(typeof(current), value)
                update = NamedTuple{(subcontext_name,)}((NamedTuple{(varname,)}((converted,)),))
                StatefulAlgorithms.context(process, StatefulAlgorithms.merge_into_subcontexts(context, update))
            end
        end
    end
    return nothing
end

function _set_temperature!(g, value)
    return settemp!(g, value)
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
    data = StatefulAlgorithms.getdata(sc)
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

function _kinetic_time_snapshot(process::StatefulAlgorithms.AbstractProcess)
    context = try
        StatefulAlgorithms.getcontext(process)
    catch
        StatefulAlgorithms.context(process)
    end
    context isa StatefulAlgorithms.ProcessContext || return nothing

    subcontexts = StatefulAlgorithms.get_subcontexts(context)
    for subcontext_name in reverse(propertynames(subcontexts))
        subcontext_name === :globals && continue
        subcontext_name === :_injector && continue
        snapshot = _kinetic_time_snapshot(getproperty(subcontexts, subcontext_name))
        isnothing(snapshot) || return snapshot
    end
    return nothing
end

function _kinetic_time_snapshot(g::IsingGraph)
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
        total += Int(StatefulAlgorithms.getticks(process))
    end
    return total
end

function _pause_graph_processes!(g)
    for process in processes(g)
        if StatefulAlgorithms.isrunning(process)
            StatefulAlgorithms.pause(process)
        end
    end
    return nothing
end

function _resume_graph_processes!(g)
    for process in processes(g)
        if StatefulAlgorithms.ispaused(process)
            run(process)
        end
    end
    return nothing
end

function _graph_paused(g)
    graph_processes = processes(g)
    isempty(graph_processes) && return false
    any(StatefulAlgorithms.isrunning, graph_processes) && return false
    return any(StatefulAlgorithms.ispaused, graph_processes)
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

function _request_process_close!(process::StatefulAlgorithms.AbstractProcess)
    try
        if applicable(StatefulAlgorithms.shouldrun, process, false)
            StatefulAlgorithms.shouldrun(process, false)
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
                    any_running |= StatefulAlgorithms.isrunning(process)
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

_magnetization(g, layer_idx) = _with_layer(layer -> sum(state(layer)), g, layer_idx)

function _layer_colorrange(layer)
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

"""
    _topology_3d_display_enabled(topology)

Return whether a three-dimensional topology should use the topology-specific
Windows display path instead of the original size-based meshscatter path.
"""
function _topology_3d_display_enabled(top::T) where {T<:AbstractLayerTopology}
    return false
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

_layer_state_view(layer) = view(state(layer), ntuple(_ -> (:), ndims(state(layer)))...)
_layer_state_vector_view(layer) = vec(_layer_state_view(layer))
_layer_state_float_vector(layer) = Float64.(vec(state(layer)))

function _cast_layer_state_vector(layer)
    layer_state = state(layer)
    unsafe_vector = unsafe_wrap(Vector{eltype(layer_state)}, pointer(layer_state), length(layer_state))
    return CastVec(Float64, unsafe_vector)
end
