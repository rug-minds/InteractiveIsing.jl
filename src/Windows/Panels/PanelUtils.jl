@inline _layer_index(layer_idx::Integer) = layer_idx
@inline _layer_index(layer_idx) = layer_idx[]

@inline _with_layer(f, g, layer_idx) =
    inline_layer_dispatch(f, _layer_index(layer_idx), layers(g))

current_layer(g, layer_idx) = _with_layer(identity, g, layer_idx)

_has_layer_selector(g) = length(layers(g)) > 1
_has_layer_selector(g::SingleLayerGraph) = false

function _set_slider_close!(slider, value)
    try
        set_close_to!(slider, value)
    catch
        slider.value[] = value
    end
    return slider
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
        try
            Processes.shouldrun(process, false)
        catch err
            @warn "Could not request process stop" process exception = (err, catch_backtrace())
        end
    end

    empty!(processes(g))
    _reap_graph_processes_later!(graph_processes)

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

function _cast_layer_state_vector(layer)
    layer_state = state(layer)
    unsafe_vector = unsafe_wrap(Vector{eltype(layer_state)}, pointer(layer_state), length(layer_state))
    return CastVec(Float64, unsafe_vector)
end
