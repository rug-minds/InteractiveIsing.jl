"""
    AbstractPanel

Supertype for mountable Windows UI components. A panel value describes what
should be mounted; a `PanelHandle` stores the resources and runtime data
created by mounting it.

Custom panels extend `mount!`.
"""
abstract type AbstractPanel end

"""
    PanelHandle

Runtime object returned by `panel!`. It stores the mounted panel, its
host, its Makie layout cell, child panels, registered resources, and arbitrary
panel data accessible as `handle[:key]`.
"""
mutable struct PanelHandle
    panel::AbstractPanel
    host::Any
    layout::Any
    slot::Any
    resources::Vector{Any}
    close_callbacks::Vector{Any}
    children::OrderedDict{Any, PanelHandle}
    data::Dict{Symbol, Any}
    closed::Bool
end

PanelHandle(panel::AbstractPanel, host, layout) =
    PanelHandle(panel, host, layout, nothing, Any[], Any[], OrderedDict{Any, PanelHandle}(), Dict{Symbol, Any}(), false)

PanelHandle(panel::AbstractPanel, host, layout, slot) =
    PanelHandle(panel, host, layout, slot, Any[], Any[], OrderedDict{Any, PanelHandle}(), Dict{Symbol, Any}(), false)

"""
    WindowHost

Root of a Windows UI. A host owns the Makie figure/screen, frame timer, polling
timer, mounted panel tree, registered resources, and window lifecycle state.
"""
mutable struct WindowHost
    uuid::UUID
    figure::Figure
    screen::Any
    open::Observable{Bool}
    fps::Float64
    polling_rate::Float64
    frame_timer::Union{PTimer, Nothing}
    poll_timer::Union{PTimer, Nothing}
    frame_callbacks::Vector{Function}
    pollables::Vector{PolledObservable}
    resources::Vector{Any}
    close_callbacks::Vector{Any}
    children::OrderedDict{Any, PanelHandle}
    paused::Observable{Bool}
    data::Dict{Symbol, Any}
    closing::Bool
    closed::Bool
end

"""
    WindowHost(fig::Figure; screen = nothing, fps = 30, polling_rate = 10,
               open = Observable(true))

Create a host around an existing Makie figure. This is useful for tests and for
embedding panels without opening a GLMakie window.
"""
function WindowHost(fig::Figure; screen = nothing, fps = 30, polling_rate = 10, open = Observable(true), start_timers = true)
    host = WindowHost(
        uuid1(),
        fig,
        screen,
        open,
        Float64(fps),
        Float64(polling_rate),
        nothing,
        nothing,
        Function[],
        PolledObservable[],
        Any[],
        Any[],
        OrderedDict{Any, PanelHandle}(),
        Observable(false),
        Dict{Symbol, Any}(),
        false,
        false,
    )
    start_timers && _start_host_timers!(host)
    return host
end

function _start_host_timers!(host::WindowHost)
    isnothing(host.frame_timer) || close(host.frame_timer)
    isnothing(host.poll_timer) || close(host.poll_timer)
    host.frame_timer = PTimer(_ -> _tick!(host), 0.; interval = 1 / host.fps)
    host.poll_timer = PTimer(_ -> _poll!(host), 0.; interval = 1 / host.polling_rate)
    return host
end

function _close_notification_timer!(timer; wait_for_close = true)
    isnothing(timer) && return nothing
    try
        if timer isa PTimer
            close(timer)
        elseif applicable(close, timer)
            close(timer)
        end
        if wait_for_close
            try
                wait(timer)
            catch
            end
        end
    catch err
        @warn "Could not stop window notification timer" timer_type = typeof(timer) exception = (err, catch_backtrace())
    end
    return nothing
end

function _stop_host_notification_timers!(host::WindowHost; wait_for_timers = true)
    frame_timer = host.frame_timer
    poll_timer = host.poll_timer
    _close_notification_timer!(frame_timer; wait_for_close = wait_for_timers)
    _close_notification_timer!(poll_timer; wait_for_close = wait_for_timers)
    host.frame_timer = nothing
    host.poll_timer = nothing
    return (frame_timer, poll_timer)
end

function _wait_notification_timers!(timers)
    for timer in timers
        isnothing(timer) && continue
        try
            wait(timer)
        catch
        end
    end
    return nothing
end

function _stop_host_notifications!(host::WindowHost; wait_for_timers = true)
    _stop_host_notification_timers!(host; wait_for_timers)

    for pollable in copy(host.pollables)
        _shutdown_runtime_resource(pollable)
    end
    for resource in reverse(host.resources)
        _shutdown_runtime_resource(resource)
    end
    for child in reverse(collect(values(host.children)))
        _shutdown_handle_runtime!(child)
    end

    empty!(host.frame_callbacks)
    empty!(host.pollables)
    return nothing
end

function _display_host!(host::WindowHost, title; focus = true)
    _clear_glmakie_screen_reuse_pool!()
    screen = GLMakie.Screen(; focus_on_show = focus)
    _disable_glmakie_screen_reuse!(screen)
    display(screen, host.figure)
    GLFW.SetWindowTitle(to_native(screen), title)
    _disable_glmakie_renderloop_close!(screen)
    _focus_native_window!(screen)
    host.screen = screen
    host.open = events(host.figure).window_open
    register!(host, on(host.open) do isopen
        isopen && return nothing
        _schedule_native_close!(host)
        return nothing
    end)
    register!(host, on(events(host.figure.scene).keyboardbutton) do _
        if ispressed(host.figure, (Keyboard.left_super, Keyboard.w)) || ispressed(host.figure, (Keyboard.left_control, Keyboard.w))
            _request_deferred_window_close!(host)
        end
    end)
    _start_host_timers!(host)
    return host
end

function _clear_glmakie_screen_reuse_pool!()
    try
        if isdefined(GLMakie, :SCREEN_REUSE_POOL)
            empty!(getfield(GLMakie, :SCREEN_REUSE_POOL))
        end
    catch err
        @warn "Could not clear GLMakie screen reuse pool" exception = (err, catch_backtrace())
    end
    return nothing
end

function _disable_glmakie_screen_reuse!(screen)
    isnothing(screen) && return nothing
    try
        if hasproperty(screen, :reuse)
            screen.reuse = false
        end
    catch err
        @warn "Could not disable GLMakie screen reuse" exception = (err, catch_backtrace())
    end
    return nothing
end

function _focus_native_window!(screen)
    isnothing(screen) && return nothing
    window = try
        to_native(screen)
    catch
        return nothing
    end
    try
        GLFW.ShowWindow(window)
    catch
    end
    try
        GLFW.RequestWindowAttention(window)
    catch
    end
    try
        GLFW.PollEvents()
    catch
    end
    return nothing
end

function _request_deferred_window_close!(host::WindowHost)
    (host.closed || host.closing) && return nothing
    host.open[] = false
    return nothing
end

function _set_glmakie_renderloop_close!(screen, value::Bool)
    isnothing(screen) && return nothing
    try
        if hasproperty(screen, :close_after_renderloop)
            screen.close_after_renderloop = value
        end
    catch err
        @warn "Could not update GLMakie renderloop close mode" exception = (err, catch_backtrace())
    end
    return nothing
end

_disable_glmakie_renderloop_close!(screen) = _set_glmakie_renderloop_close!(screen, false)

function _close_glmakie_screen_after_runtime_stop!(screen)
    isnothing(screen) && return nothing
    try
        close(screen; reuse = false)
    catch err
        @warn "Could not close GLMakie screen" exception = (err, catch_backtrace())
    end
    return nothing
end

"""
    window(; title = "Interactive Ising Simulation", size = (1500, 1500),
             fps = 30, polling_rate = 10, kwargs...) -> WindowHost

Open a GLMakie screen, display a new figure, and return the owning
`WindowHost`.
"""
function window(; title = "Interactive Ising Simulation", size = (1500, 1500), fps = 30, polling_rate = 10, focus = true, kwargs...)
    fig = Figure(; size, kwargs...)
    host = WindowHost(fig; screen = nothing, fps, polling_rate, open = Observable(true), start_timers = false)
    host.data[:title] = title
    return _display_host!(host, title; focus)
end

Base.getindex(host::WindowHost, key::Symbol) = host.data[key]
Base.setindex!(host::WindowHost, value, key::Symbol) = setindex!(host.data, value, key)
Base.haskey(host::WindowHost, key::Symbol) = haskey(host.data, key)

Base.getindex(handle::PanelHandle, key::Symbol) = handle.data[key]
Base.setindex!(handle::PanelHandle, value, key::Symbol) = setindex!(handle.data, value, key)
Base.haskey(handle::PanelHandle, key::Symbol) = haskey(handle.data, key)

_typename(value) = nameof(typeof(value))
_window_title(host::WindowHost) = get(host.data, :title, "untitled")

# TODO: Move shows into separate file
function Base.show(io::IO, panel::AbstractPanel)
    print(io, _typename(panel), "(...)")
end

Base.show(io::IO, ::MIME"text/plain", panel::AbstractPanel) = show(io, panel)

function Base.show(io::IO, handle::PanelHandle)
    print(
        io,
        "PanelHandle(",
        _typename(handle.panel),
        ", slot=",
        handle.slot,
        ", children=",
        length(handle.children),
        ", resources=",
        length(handle.resources),
        ", close_callbacks=",
        length(handle.close_callbacks),
        ", closed=",
        handle.closed,
        ")",
    )
end

Base.show(io::IO, ::MIME"text/plain", handle::PanelHandle) = show(io, handle)

function Base.show(io::IO, host::WindowHost)
    print(
        io,
        "WindowHost(",
        repr(_window_title(host)),
        ", panels=",
        length(host.children),
        ", resources=",
        length(host.resources),
        ", close_callbacks=",
        length(host.close_callbacks),
        ", fps=",
        host.fps,
        ", polling_rate=",
        host.polling_rate,
        ", closed=",
        host.closed,
        ")",
    )
end

Base.show(io::IO, ::MIME"text/plain", host::WindowHost) = show(io, host)

function _grid_cell(parent, pos)
    pos isa Tuple && return parent[pos...]
    return pos
end

_child_key(pos; key = nothing) = isnothing(key) ? pos : key

function _store_child!(children::OrderedDict{Any, PanelHandle}, key, handle::PanelHandle)
    if haskey(children, key)
        close(children[key])
    end
    children[key] = handle
    return handle
end

"""
    panel!(host, panel, pos = (1, 1); key = nothing, kwargs...) -> PanelHandle
    panel!(parent_handle, panel, pos; key = nothing, kwargs...) -> PanelHandle

Mount an `AbstractPanel` into a host or parent panel. `pos` can be a
Makie grid position tuple or an already selected layout cell. If `key` is
provided, the child handle is stored under that key; otherwise `pos` is used.
"""
function panel!(host::WindowHost, panel::AbstractPanel, pos = (1, 1); key = nothing, kwargs...)
    cell = _grid_cell(host.figure, pos)
    handle = mount!(panel, host, cell; kwargs...)
    handle.slot = pos
    return _store_child!(host.children, _child_key(pos; key), handle)
end

panel!(host::WindowHost, key, panel::AbstractPanel, pos = (1, 1); kwargs...) =
    panel!(host, panel, pos; key, kwargs...)

function panel!(parent::PanelHandle, panel::AbstractPanel, pos; key = nothing, kwargs...)
    cell = _grid_cell(parent.layout, pos)
    handle = mount!(panel, parent.host, cell; kwargs...)
    handle.slot = pos
    return _store_child!(parent.children, _child_key(pos; key), handle)
end

panel!(parent::PanelHandle, key, panel::AbstractPanel, pos; kwargs...) =
    panel!(parent, panel, pos; key, kwargs...)

"""
    mount!(panel::AbstractPanel, host::WindowHost, cell; kwargs...) -> PanelHandle

Panel construction hook. Implement this for custom panel types. The method
should populate `cell`, register resources/callbacks, and return a
`PanelHandle`.
"""
mount!(panel::AbstractPanel, host::WindowHost, cell; kwargs...) =
    error("No Windows.mount! method defined for $(typeof(panel)).")

"""
    register!(host_or_handle, resource)

Register `resource` for lifecycle cleanup. Observer functions, timers, polled
observables, processes, and objects with `close(resource)` are cleaned up when
the owning host or panel closes.
"""
function register!(host::WindowHost, resource)
    push!(host.resources, resource)
    return resource
end

function register!(handle::PanelHandle, resource)
    push!(handle.resources, resource)
    return resource
end

struct HotObservable
    observable::Any
end

"""
    register_hot_observable!(host_or_handle, observable)

Register a high-frequency Makie observable whose value may point at live
simulation memory. During runtime shutdown the observable's stored value is
replaced with an inert zero-sized value derived from the observable type
parameter, without calling `notify`.
"""
function register_hot_observable!(host::WindowHost, observable::Observable)
    register!(host, HotObservable(observable))
    return observable
end

function register_hot_observable!(handle::PanelHandle, observable::Observable)
    register!(handle, HotObservable(observable))
    return observable
end

"""
    hot_observable_zero(::Type{T})
    hot_observable_zero(observable::Observable{T})

Allocate an inert zero-sized replacement suitable for an observable value of
type `T`. This is used during close cleanup to sever references from Makie plots
to hot simulation memory without issuing another observable notification.
"""
hot_observable_zero(observable::Observable{T}) where {T} = hot_observable_zero(T)

function hot_observable_zero(::Type{T}) where {E,N,T<:AbstractArray{E,N}}
    return zeros(E, ntuple(_ -> 0, N))
end

function hot_observable_zero(::Type{T}) where {E,N,T<:Array{E,N}}
    return zeros(E, ntuple(_ -> 0, N))
end

function hot_observable_zero(::Type{T}) where {E,N,P,I,L,T<:SubArray{E,N,P,I,L}}
    replacement = view(zeros(E, ntuple(_ -> 0, N)), ntuple(_ -> (:), N)...)
    replacement isa T && return replacement
    throw(ArgumentError("Cannot build zero-sized replacement for hot observable value type $T."))
end

function hot_observable_zero(::Type{T}) where {E,N,P<:SubArray{E,N},T<:Base.ReshapedArray{E,1,P,Tuple{}}}
    replacement = vec(hot_observable_zero(P))
    replacement isa T && return replacement
    throw(ArgumentError("Cannot build zero-sized replacement for hot observable value type $T."))
end

function detach_hot_observable!(observable::Observable{T}) where {T}
    replacement = hot_observable_zero(T)
    replacement isa T || throw(ArgumentError("Replacement value $(typeof(replacement)) is not compatible with observable value type $T."))
    setfield!(observable, :val, replacement)
    return observable
end

struct CloseCallback
    callback::Function
end

"""
    onclose!(host_or_handle, callback)

Register `callback(owner)` to run once when a window host or mounted panel is
closed. Close callbacks run for both explicit `close(host)` calls and native
window close events.

Callbacks are scheduled asynchronously and do not block native GLMakie window
teardown.
"""
function onclose!(host::WindowHost, callback::Function)
    wrapped = CloseCallback(callback)
    push!(host.close_callbacks, wrapped)
    return wrapped
end
onclose!(callback::Function, host::WindowHost) = onclose!(host, callback)

function onclose!(handle::PanelHandle, callback::Function)
    wrapped = CloseCallback(callback)
    push!(handle.close_callbacks, wrapped)
    return wrapped
end
onclose!(callback::Function, handle::PanelHandle) = onclose!(handle, callback)

"""
    register_frame!(host_or_handle, callback)

Run `callback(host)` on each frame tick. Frame callbacks registered through a
panel handle are removed automatically when that panel closes.
"""
function register_frame!(host::WindowHost, callback::Function)
    push!(host.frame_callbacks, callback)
    return callback
end
register_frame!(callback::Function, host::WindowHost) = register_frame!(host, callback)

function register_frame!(handle::PanelHandle, callback::Function)
    register_frame!(handle.host, callback)
    register!(handle, FrameCallback(handle.host, callback))
    return callback
end
register_frame!(callback::Function, handle::PanelHandle) = register_frame!(handle, callback)

"""
    register_polled!(host_or_handle, po::PolledObservable)

Register a `PolledObservable` with the host polling timer. Polled
observables registered through a panel handle are removed and closed with that
panel.
"""
function register_polled!(host::WindowHost, po::PolledObservable)
    push!(host.pollables, po)
    register!(host, po)
    return po
end

function register_polled!(handle::PanelHandle, po::PolledObservable)
    push!(handle.host.pollables, po)
    register!(handle, PolledRegistration(handle.host, po))
    return po
end

struct FrameCallback
    host::WindowHost
    callback::Function
end

struct PolledRegistration
    host::WindowHost
    observable::PolledObservable
end

function _tick!(host::WindowHost)
    (host.closed || host.closing) && return nothing
    if !host.open[]
        _schedule_native_close!(host)
        return nothing
    end
    for callback in copy(host.frame_callbacks)
        (host.closed || host.closing) && return nothing
        callback(host)
    end
    return nothing
end

function _poll!(host::WindowHost)
    (host.closed || host.closing) && return nothing
    if !host.open[]
        _schedule_native_close!(host)
        return nothing
    end
    for observable in copy(host.pollables)
        (host.closed || host.closing) && return nothing
        poll!(observable)
    end
    return nothing
end

function _cleanup_resource(resource)
    try
        _cleanup_resource!(resource)
    catch err
        @warn "Error while closing window resource" resource_type = typeof(resource) exception = (err, catch_backtrace())
    end
    return nothing
end

_cleanup_resource!(::Nothing) = nothing

# Does this one make sense?
_cleanup_resource!(callback::FrameCallback) = filter!(!isequal(callback.callback), callback.host.frame_callbacks)
function _cleanup_resource!(resource::HotObservable)
    detach_hot_observable!(resource.observable)
    return nothing
end
function _cleanup_resource!(registration::PolledRegistration)
    filter!(!isequal(registration.observable), registration.host.pollables)
    close(registration.observable)
    return nothing
end
# What does it mean to clean a callback?
_cleanup_resource!(callback::Function) = callback()
_cleanup_resource!(observer::Observables.ObserverFunction) = off(observer)
_cleanup_resource!(timer::PTimer) = close(timer)
_cleanup_resource!(timer::Timer) = close(timer)
_cleanup_resource!(po::PolledObservable) = close(po)
function _cleanup_resource!(process::Processes.AbstractProcess)
    _request_process_close!(process)
    return nothing
end
_cleanup_resource!(resource) = applicable(close, resource) ? close(resource) : nothing

function _cleanup_native_resource(resource)
    try
        _cleanup_native_resource!(resource)
    catch err
        @warn "Error while detaching native window resource" resource_type = typeof(resource) exception = (err, catch_backtrace())
    end
    return nothing
end

_cleanup_native_resource!(::Nothing) = nothing
_cleanup_native_resource!(callback::FrameCallback) = filter!(!isequal(callback.callback), callback.host.frame_callbacks)
function _cleanup_native_resource!(registration::PolledRegistration)
    filter!(!isequal(registration.observable), registration.host.pollables)
    close(registration.observable)
    return nothing
end
_cleanup_native_resource!(observer::Observables.ObserverFunction) = off(observer)
_cleanup_native_resource!(timer::PTimer) = close(timer)
_cleanup_native_resource!(timer::Timer) = close(timer)
_cleanup_native_resource!(po::PolledObservable) = close(po)
_cleanup_native_resource!(process::Processes.AbstractProcess) = _cleanup_resource!(process)
_cleanup_native_resource!(resource) = nothing

function _shutdown_runtime_resource(resource)
    try
        _shutdown_runtime_resource!(resource)
    catch err
        @warn "Error while stopping window runtime resource" resource_type = typeof(resource) exception = (err, catch_backtrace())
    end
    return nothing
end

_shutdown_runtime_resource!(::Nothing) = nothing
_shutdown_runtime_resource!(::FrameCallback) = nothing
function _shutdown_runtime_resource!(resource::HotObservable)
    detach_hot_observable!(resource.observable)
    return nothing
end
function _shutdown_runtime_resource!(registration::PolledRegistration)
    close(registration.observable)
    return nothing
end
_shutdown_runtime_resource!(::Observables.ObserverFunction) = nothing
_shutdown_runtime_resource!(timer::PTimer) = close(timer)
_shutdown_runtime_resource!(timer::Timer) = close(timer)
_shutdown_runtime_resource!(po::PolledObservable) = close(po)
function _shutdown_runtime_resource!(process::Processes.AbstractProcess)
    _request_process_close!(process)
    return nothing
end
_shutdown_runtime_resource!(resource) = nothing

function _request_host_owned_process_shutdown!(host::WindowHost)
    close_graphs = get(host.data, :close_graphs, nothing)
    if close_graphs !== nothing
        for g in collect(keys(close_graphs))
            _request_graph_process_close!(g)
        end
    end

    close_processes = get(host.data, :close_processes, nothing)
    if close_processes !== nothing
        for process in collect(keys(close_processes))
            _request_process_close!(process)
        end
    end
    return nothing
end

function _run_close_callback(owner, callback::CloseCallback)
    @async try
        callback.callback(owner)
    catch err
        @warn "Error in window close callback" owner_type = typeof(owner) exception = (err, catch_backtrace())
    end
    return nothing
end

function _run_close_callbacks!(owner)
    callbacks = reverse(copy(getfield(owner, :close_callbacks)))
    empty!(getfield(owner, :close_callbacks))
    for callback in callbacks
        _run_close_callback(owner, callback)
    end
    return nothing
end

function _pause_resource(resource)
    try
        if resource isa PTimer
            close(resource)
        elseif resource isa PolledObservable
            close(resource)
        elseif resource isa Processes.AbstractProcess
            Processes.pause(resource)
        end
    catch err
        @warn "Error while pausing window resource" resource_type = typeof(resource) exception = (err, catch_backtrace())
    end
    return nothing
end

function _resume_resource(resource)
    try
        if resource isa PTimer
            Processes.start(resource)
        elseif resource isa PolledObservable
            Processes.start(resource)
        elseif resource isa Processes.AbstractProcess
            run(resource)
        end
    catch err
        @warn "Error while resuming window resource" resource_type = typeof(resource) exception = (err, catch_backtrace())
    end
    return nothing
end
## What are these?
close!(panel::AbstractPanel, handle::PanelHandle) = nothing
pause!(panel::AbstractPanel, handle::PanelHandle) = nothing
resume!(panel::AbstractPanel, handle::PanelHandle) = nothing
restart!(panel::AbstractPanel, handle::PanelHandle) = nothing

"""
    close(handle::PanelHandle)

Close a mounted panel once. Children are closed first, then the panel-specific
`close!(panel, handle)` hook runs, then registered resources are cleaned up.
"""
function Base.close(handle::PanelHandle)
    handle.closed && return nothing
    handle.closed = true
    for child in reverse(collect(values(handle.children)))
        close(child)
    end
    close!(handle.panel, handle)
    _run_close_callbacks!(handle)
    for resource in reverse(handle.resources)
        _cleanup_resource(resource)
    end
    empty!(handle.resources)
    empty!(handle.children)
    return nothing
end

function _mark_closed!(handle::PanelHandle)
    handle.closed = true
    for child in values(handle.children)
        _mark_closed!(child)
    end
    empty!(handle.resources)
    empty!(handle.close_callbacks)
    empty!(handle.children)
    return nothing
end

function _shutdown_handle_runtime!(handle::PanelHandle)
    handle.closed && return nothing
    handle.closed = true
    for child in reverse(collect(values(handle.children)))
        _shutdown_handle_runtime!(child)
    end
    for resource in reverse(handle.resources)
        _shutdown_runtime_resource(resource)
    end
    return nothing
end

function _begin_native_close!(host::WindowHost)
    (host.closed || host.closing) && return nothing
    host.closing = true
    _stop_host_notifications!(host; wait_for_timers = true)
    _request_host_owned_process_shutdown!(host)
    host.closed = true
    return host
end

function _schedule_native_close!(host::WindowHost)
    (host.closed || get(host.data, :close_scheduled, false)) && return nothing
    host.data[:close_scheduled] = true
    host.closing = true
    screen = host.screen
    @async begin
        _stop_host_notifications!(host; wait_for_timers = true)
        _request_host_owned_process_shutdown!(host)
        _close_glmakie_screen_after_runtime_stop!(screen)
        host.closed = true
        _finish_native_close!(host)
    end
    return nothing
end

function _finish_native_close!(host::WindowHost)
    _run_close_callbacks!(host)
    host.closing = false
    return nothing
end

"""
    close(host::WindowHost)

Close a window host once. Frame and polling timers stop before child panels are
closed, so callbacks cannot keep notifying Makie objects during native window
teardown.
"""
function Base.close(host::WindowHost)
    (host.closed || host.closing) && return nothing
    if !isnothing(host.screen)
        host.open[] = false
        return nothing
    end

    host.closing = true
    _stop_host_notifications!(host; wait_for_timers = true)
    _request_host_owned_process_shutdown!(host)
    _run_close_callbacks!(host)
    host.closed = true
    host.closing = false
    return nothing
end

"""
    pause!(handle_or_host)

Propagate a pause lifecycle event through panel resources and children. Pausing
a host does not stop the host frame or polling timers; the interface keeps
updating while graph processes are paused.
"""
function pause!(handle::PanelHandle)
    pause!(handle.panel, handle)
    _pause_resource.(handle.resources)
    pause!.(values(handle.children))
    return handle
end

"""
    resume!(handle_or_host)

Resume resources and children paused through `pause!`.
"""
function resume!(handle::PanelHandle)
    resume!(handle.panel, handle)
    _resume_resource.(handle.resources)
    resume!.(values(handle.children))
    return handle
end

"""
    restart!(handle)

Run the panel-specific restart hook. The default hook is a no-op; panels that
own restartable processes can extend `restart!(panel, handle)`.
"""
function restart!(handle::PanelHandle)
    restart!(handle.panel, handle)
    return handle
end

function pause!(host::WindowHost)
    host.paused[] && return host
    host.paused[] = true
    pause!.(values(host.children))
    return host
end

function resume!(host::WindowHost)
    !host.paused[] && return host
    host.paused[] = false
    resume!.(values(host.children))
    return host
end
