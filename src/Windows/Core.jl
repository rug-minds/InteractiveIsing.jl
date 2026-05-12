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

function _display_host!(host::WindowHost, title)
    screen = GLMakie.Screen()
    display(screen, host.figure)
    GLFW.SetWindowTitle(to_native(screen), title)
    _disable_glmakie_renderloop_close!(screen)
    host.screen = screen
    host.open = events(host.figure).window_open
    register!(host, on(events(host.figure.scene).keyboardbutton) do _
        if ispressed(host.figure, (Keyboard.left_super, Keyboard.w)) || ispressed(host.figure, (Keyboard.left_control, Keyboard.w))
            close(host)
        end
    end)
    _start_host_timers!(host)
    return host
end

function _disable_glmakie_renderloop_close!(screen)
    isnothing(screen) && return nothing
    try
        if hasproperty(screen, :close_after_renderloop)
            screen.close_after_renderloop = false
        end
    catch err
        @warn "Could not disable GLMakie renderloop close" exception = (err, catch_backtrace())
    end
    return nothing
end

function _stop_glmakie_renderloop_without_wait!(screen)
    isnothing(screen) && return nothing
    _disable_glmakie_renderloop_close!(screen)
    try
        if hasproperty(screen, :stop_renderloop)
            screen.stop_renderloop[] = true
        end
    catch err
        @warn "Could not request GLMakie renderloop stop" exception = (err, catch_backtrace())
    end
    return nothing
end

function _forget_glmakie_screen!(screen)
    isnothing(screen) && return nothing
    try
        isdefined(GLMakie, :ALL_SCREENS) && delete!(getfield(GLMakie, :ALL_SCREENS), screen)
        isdefined(GLMakie, :SCREEN_REUSE_POOL) && delete!(getfield(GLMakie, :SCREEN_REUSE_POOL), screen)
        if isdefined(GLMakie, :SINGLETON_SCREEN)
            singleton = getfield(GLMakie, :SINGLETON_SCREEN)
            screen in singleton && empty!(singleton)
        end
    catch err
        @warn "Could not detach GLMakie screen from closeall registry" exception = (err, catch_backtrace())
    end
    return nothing
end

function _destroy_native_glfw_window!(screen)
    isnothing(screen) && return nothing
    window = try
        to_native(screen)
    catch
        return nothing
    end

    try
        getfield(window, :handle) == C_NULL && return nothing
    catch
    end

    try
        GLFW.SetWindowCloseCallback(window, nothing)
    catch
    end
    try
        GLFW.HideWindow(window)
    catch
    end
    try
        GLFW.SetWindowShouldClose(window, true)
    catch
    end
    try
        GLFW.PollEvents()
    catch
    end
    try
        GLFW.DestroyWindow(window)
        try
            window.handle = C_NULL
        catch
        end
    catch err
        @warn "Could not destroy native GLFW window" exception = (err, catch_backtrace())
    end
    return nothing
end

function _request_native_window_close!(host::WindowHost)
    screen = host.screen
    isnothing(screen) && return nothing
    _stop_glmakie_renderloop_without_wait!(screen)
    try
        host.open[] = false
    catch
    end
    try
        GLFW.SetWindowShouldClose(to_native(screen), true)
        GLFW.PollEvents()
    catch err
        @warn "Could not request native GLMakie window close" exception = (err, catch_backtrace())
    end
    _destroy_native_glfw_window!(screen)
    _forget_glmakie_screen!(screen)
    return nothing
end

"""
    window(; title = "Interactive Ising Simulation", size = (1500, 1500),
             fps = 30, polling_rate = 10, kwargs...) -> WindowHost

Open a GLMakie screen, display a new figure, and return the owning
`WindowHost`.
"""
function window(; title = "Interactive Ising Simulation", size = (1500, 1500), fps = 30, polling_rate = 10, kwargs...)
    fig = Figure(; size, kwargs...)
    host = WindowHost(fig; screen = nothing, fps, polling_rate, open = Observable(true), start_timers = false)
    host.data[:title] = title
    return _display_host!(host, title)
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
    _disable_glmakie_renderloop_close!(host.screen)
    _forget_glmakie_screen!(host.screen)
    isnothing(host.frame_timer) || close(host.frame_timer)
    isnothing(host.poll_timer) || close(host.poll_timer)
    for pollable in copy(host.pollables)
        _shutdown_runtime_resource(pollable)
    end
    for resource in reverse(host.resources)
        _shutdown_runtime_resource(resource)
    end
    for child in reverse(collect(values(host.children)))
        _shutdown_handle_runtime!(child)
    end
    _request_host_owned_process_shutdown!(host)
    empty!(host.frame_callbacks)
    empty!(host.pollables)
    host.closed = true
    return host
end

function _schedule_native_close!(host::WindowHost)
    _begin_native_close!(host) === nothing && return nothing
    _request_native_window_close!(host)
    @async _finish_native_close!(host)
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
        _begin_native_close!(host)
        @async _finish_native_close!(host)
        _request_native_window_close!(host)
        return nothing
    end

    host.closing = true
    should_close_screen = !isnothing(host.screen) && host.open[]
    isnothing(host.frame_timer) || close(host.frame_timer)
    isnothing(host.poll_timer) || close(host.poll_timer)
    for pollable in copy(host.pollables)
        _shutdown_runtime_resource(pollable)
    end
    for resource in reverse(host.resources)
        _shutdown_runtime_resource(resource)
    end
    for child in reverse(collect(values(host.children)))
        _shutdown_handle_runtime!(child)
    end
    _request_host_owned_process_shutdown!(host)
    _run_close_callbacks!(host)
    empty!(host.frame_callbacks)
    empty!(host.pollables)
    if should_close_screen
        _request_native_window_close!(host)
    end
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
