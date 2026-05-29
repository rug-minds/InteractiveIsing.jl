@inline _layer_index(layer_idx::Integer) = layer_idx
@inline _layer_index(layer_idx) = layer_idx[]

@inline _with_layer(f, g, layer_idx) =
    inline_layer_dispatch(f, _layer_index(layer_idx), layers(g))

current_layer(g, layer_idx) = _with_layer(identity, g, layer_idx)

_has_layer_selector(g) = length(layers(g)) > 1
_has_layer_selector(g::SingleLayerGraph) = false

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
_temperature_value(_) = nothing

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

function _set_process_context_temperature!(process::Processes.AbstractProcess, value)
    context = Processes.context(process)
    context isa Processes.ProcessContext || return nothing

    subcontexts = Processes.get_subcontexts(context)
    for subcontext_name in propertynames(subcontexts)
        subcontext_name === :globals && continue
        subcontext_name === :_injector && continue

        subcontext = getproperty(subcontexts, subcontext_name)
        for (varname, current) in _temperature_vars(subcontext)
            if current isa Base.RefValue
                current[] = convert(typeof(current[]), value)
            elseif Processes.isinteractive(process)
                Processes.interact!(process, Processes.Var(subcontext_name, varname) => value)
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
