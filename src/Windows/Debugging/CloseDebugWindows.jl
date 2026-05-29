"""
    close_debug_window_names() -> Vector{Symbol}

Return the names of stripped-down close-debug windows. Open them one at a time
with `open_close_debug_window(g, name)` and close each native window manually to
isolate which facility triggers a GLMakie close freeze.
"""
function close_debug_window_names()
    return [
        :raw_glmakie_label,
        :host_no_timers,
        :host_empty,
        :host_frame_label,
        :host_polled_label,
        :static_layer,
        :layer_view,
        :all_layers_view,
        :temperature,
        :status,
        :magnetization,
        :hamiltonian,
        :simulation_hidden_left,
        :simulation_full,
        :public_interface,
        :layer_view_running_langevin,
        :all_layers_view_running_langevin,
        :status_running_langevin,
        :simulation_running_langevin,
        :public_interface_running_langevin,
        :simulation_open_then_langevin,
        :public_interface_open_then_langevin,
        :simulation_open_then_metropolis,
        :public_interface_open_then_metropolis,
        :simulation_bad_sync_close_graph,
        :simulation_async_close_graph,
        :subset_status_layer_langevin,
        :subset_layer_only_langevin,
        :all_layers_view_open_then_langevin,
        :subset_layer_only_empty_on_close_langevin,
        :subset_layer_only_copy_langevin,
        :subset_layer_only_no_polltimer_langevin,
        :subset_temperature_only_langevin,
        :subset_kinetic_only_langevin,
        :subset_layer_temperature_langevin,
        :subset_layer_temperature_no_kinetic_langevin,
        :subset_layer_kinetic_langevin,
        :subset_layer_hamiltonian_langevin,
        :subset_layer_magnetization_langevin,
        :subset_status_layer_temperature_langevin,
        :subset_full_no_status_langevin,
        :subset_full_no_temperature_langevin,
        :subset_full_no_hamiltonian_langevin,
        :subset_full_no_magnetization_langevin,
    ]
end

"""
    close_debug_window_descriptions() -> Vector{NamedTuple}

Return `(name, needs_graph, description)` entries for the close-debug windows.
"""
function close_debug_window_descriptions()
    return [
        (name = :raw_glmakie_label, needs_graph = false, description = "Plain GLMakie Screen + Figure + Label. No WindowHost, no timers, no package close path."),
        (name = :host_no_timers, needs_graph = false, description = "WindowHost close observer and Cmd+W hook, but no frame/poll timers."),
        (name = :host_empty, needs_graph = false, description = "Normal WindowHost with frame/poll timers, no panels."),
        (name = :host_frame_label, needs_graph = false, description = "Normal WindowHost; frame timer updates a Label Observable."),
        (name = :host_polled_label, needs_graph = false, description = "Normal WindowHost; polling timer updates a PolledObservable-backed Label."),
        (name = :static_layer, needs_graph = true, description = "Graph state plot once, no frame notify and no graph close registration."),
        (name = :layer_view, needs_graph = true, description = "Existing LayerViewPanel only: graph state plot, frame notify, graph close registration."),
        (name = :all_layers_view, needs_graph = true, description = "Existing AllLayersViewPanel only: all positioned layer plots, frame notify, graph close registration."),
        (name = :temperature, needs_graph = true, description = "Existing TemperaturePanel only: slider + graph-backed PolledObservable."),
        (name = :status, needs_graph = true, description = "Existing StatusPanel only: step counter, pause button, graph pause polling."),
        (name = :magnetization, needs_graph = true, description = "Existing MagnetizationPanel only: magnetization polling + textbox."),
        (name = :hamiltonian, needs_graph = true, description = "Existing HamiltonianParameterPanel only: selector buttons + midpanel display."),
        (name = :simulation_hidden_left, needs_graph = true, description = "Default SimulationPanel with Hamiltonian buttons hidden."),
        (name = :simulation_full, needs_graph = true, description = "Full default SimulationPanel, equivalent to interface(g)."),
        (name = :public_interface, needs_graph = true, description = "The exact public interface(g) entrypoint."),
        (name = :layer_view_running_langevin, needs_graph = true, description = "LayerViewPanel only, but starts a LocalLangevin process before opening."),
        (name = :all_layers_view_running_langevin, needs_graph = true, description = "AllLayersViewPanel only, but starts a LocalLangevin process before opening."),
        (name = :status_running_langevin, needs_graph = true, description = "StatusPanel only, but starts a LocalLangevin process before opening."),
        (name = :simulation_running_langevin, needs_graph = true, description = "SimulationPanel with a LocalLangevin process already running."),
        (name = :public_interface_running_langevin, needs_graph = true, description = "Exact interface(g) with a LocalLangevin process already running."),
        (name = :simulation_open_then_langevin, needs_graph = true, description = "SimulationPanel first, then starts LocalLangevin after the window is displayed."),
        (name = :public_interface_open_then_langevin, needs_graph = true, description = "Exact interface(g), then starts LocalLangevin after the window is displayed."),
        (name = :simulation_open_then_metropolis, needs_graph = true, description = "SimulationPanel first, then starts the default Metropolis process after display."),
        (name = :public_interface_open_then_metropolis, needs_graph = true, description = "Exact interface(g), then starts the default Metropolis process after display."),
        (name = :simulation_bad_sync_close_graph, needs_graph = true, description = "SimulationPanel plus a deliberately bad synchronous onclose close(g) callback."),
        (name = :simulation_async_close_graph, needs_graph = true, description = "SimulationPanel plus an explicit async close(g) callback."),
        (name = :subset_status_layer_langevin, needs_graph = true, description = "Subset composite: StatusPanel + LayerViewPanel, then starts a graph process (LocalLangevin by default)."),
        (name = :subset_layer_only_langevin, needs_graph = true, description = "Subset composite: LayerViewPanel only, then starts a graph process (LocalLangevin by default)."),
        (name = :all_layers_view_open_then_langevin, needs_graph = true, description = "AllLayersViewPanel only, then starts a graph process. This is the closest generic suspect for MNIST-style all-layer windows."),
        (name = :subset_layer_only_empty_on_close_langevin, needs_graph = true, description = "Layer panel only, then starts a graph process. During shutdown it swaps the plot observable value to an empty array before GLMakie close."),
        (name = :subset_layer_only_copy_langevin, needs_graph = true, description = "Copy-buffer layer panel only, then starts a graph process. Tests whether decoupling Makie from live graph state fixes close freezes."),
        (name = :subset_layer_only_no_polltimer_langevin, needs_graph = true, description = "LayerViewPanel only with host poll timer disabled, then starts a graph process (LocalLangevin by default)."),
        (name = :subset_temperature_only_langevin, needs_graph = true, description = "Subset composite: TemperaturePanel only, then starts a graph process (LocalLangevin by default)."),
        (name = :subset_kinetic_only_langevin, needs_graph = true, description = "Subset composite: KineticTimePanel only, then starts a graph process (LocalLangevin by default)."),
        (name = :subset_layer_temperature_langevin, needs_graph = true, description = "Subset composite: LayerViewPanel + KineticTimePanel + TemperaturePanel, then starts a graph process (LocalLangevin by default)."),
        (name = :subset_layer_temperature_no_kinetic_langevin, needs_graph = true, description = "Subset composite: LayerViewPanel + TemperaturePanel, no KineticTimePanel, then starts a graph process (LocalLangevin by default)."),
        (name = :subset_layer_kinetic_langevin, needs_graph = true, description = "Subset composite: LayerViewPanel + KineticTimePanel, no TemperaturePanel, then starts a graph process (LocalLangevin by default)."),
        (name = :subset_layer_hamiltonian_langevin, needs_graph = true, description = "Subset composite: HamiltonianParameterPanel only, then starts a graph process (LocalLangevin by default)."),
        (name = :subset_layer_magnetization_langevin, needs_graph = true, description = "Subset composite: LayerViewPanel + MagnetizationPanel, then starts a graph process (LocalLangevin by default)."),
        (name = :subset_status_layer_temperature_langevin, needs_graph = true, description = "Subset composite: StatusPanel + LayerViewPanel + TemperaturePanel, then starts a graph process (LocalLangevin by default)."),
        (name = :subset_full_no_status_langevin, needs_graph = true, description = "Simulation-like composite without StatusPanel, then starts a graph process (LocalLangevin by default)."),
        (name = :subset_full_no_temperature_langevin, needs_graph = true, description = "Simulation-like composite without TemperaturePanel, then starts a graph process (LocalLangevin by default)."),
        (name = :subset_full_no_hamiltonian_langevin, needs_graph = true, description = "Simulation-like composite without HamiltonianParameterPanel, then starts a graph process (LocalLangevin by default)."),
        (name = :subset_full_no_magnetization_langevin, needs_graph = true, description = "Simulation-like composite without MagnetizationPanel, then starts a graph process (LocalLangevin by default)."),
    ]
end

"""
    open_close_debug_window(g, name::Symbol; kwargs...)
    open_close_debug_window(name::Symbol; kwargs...)

Open one stripped-down close-debug window. Graph-less variants may be opened
without `g`; graph variants require an `IsingGraph`.
"""
function open_close_debug_window(g, name::Symbol; kwargs...)
    name === :raw_glmakie_label && return _debug_raw_glmakie_label(; kwargs...)
    name === :host_no_timers && return _debug_host_no_timers(; kwargs...)
    name === :host_empty && return _debug_host_empty(; kwargs...)
    name === :host_frame_label && return _debug_host_frame_label(; kwargs...)
    name === :host_polled_label && return _debug_host_polled_label(; kwargs...)

    isnothing(g) && throw(ArgumentError("close-debug window $name requires a graph."))
    name === :static_layer && return _debug_static_layer(g; kwargs...)
    name === :layer_view && return _debug_layer_view(g; kwargs...)
    name === :all_layers_view && return _debug_all_layers_view(g; kwargs...)
    name === :temperature && return _debug_temperature(g; kwargs...)
    name === :status && return _debug_status(g; kwargs...)
    name === :magnetization && return _debug_magnetization(g; kwargs...)
    name === :hamiltonian && return _debug_hamiltonian(g; kwargs...)
    name === :simulation_hidden_left && return _debug_simulation(g; hide_left_buttons = true, kwargs...)
    name === :simulation_full && return _debug_simulation(g; hide_left_buttons = false, kwargs...)
    name === :public_interface && return _debug_public_interface(g; kwargs...)
    name === :layer_view_running_langevin && return _debug_with_process(g, _debug_layer_view; kwargs...)
    name === :all_layers_view_running_langevin && return _debug_with_process(g, _debug_all_layers_view; kwargs...)
    name === :status_running_langevin && return _debug_with_process(g, _debug_status; kwargs...)
    name === :simulation_running_langevin && return _debug_with_process(g, gg -> _debug_simulation(gg; hide_left_buttons = false); kwargs...)
    name === :public_interface_running_langevin && return _debug_with_process(g, _debug_public_interface; kwargs...)
    name === :simulation_open_then_langevin && return _debug_open_then_process(g, gg -> _debug_simulation(gg; hide_left_buttons = false), _debug_langevin_algorithm(); kwargs...)
    name === :public_interface_open_then_langevin && return _debug_open_then_process(g, _debug_public_interface, _debug_langevin_algorithm(); kwargs...)
    name === :simulation_open_then_metropolis && return _debug_open_then_process(g, gg -> _debug_simulation(gg; hide_left_buttons = false), nothing; kwargs...)
    name === :public_interface_open_then_metropolis && return _debug_open_then_process(g, _debug_public_interface, nothing; kwargs...)
    name === :simulation_bad_sync_close_graph && return _debug_simulation_sync_close_graph(g; kwargs...)
    name === :simulation_async_close_graph && return _debug_simulation_async_close_graph(g; kwargs...)
    name === :subset_status_layer_langevin && return _debug_subset_then_langevin(g; status = true, layer = true, kinetic = false, temperature = false, hamiltonian = false, magnetization = false, kwargs...)
    name === :subset_layer_only_langevin && return _debug_subset_then_langevin(g; status = false, layer = true, kinetic = false, temperature = false, hamiltonian = false, magnetization = false, kwargs...)
    name === :all_layers_view_open_then_langevin && return _debug_open_then_process(g, _debug_all_layers_view, _debug_langevin_algorithm(); kwargs...)
    name === :subset_layer_only_empty_on_close_langevin && return _debug_open_then_process(g, _debug_empty_on_close_layer_view, _debug_langevin_algorithm(); kwargs...)
    name === :subset_layer_only_copy_langevin && return _debug_open_then_process(g, _debug_copy_layer_view, _debug_langevin_algorithm(); kwargs...)
    name === :subset_layer_only_no_polltimer_langevin && return _debug_subset_then_langevin(g; status = false, layer = true, kinetic = false, temperature = false, hamiltonian = false, magnetization = false, start_poll_timer = false, kwargs...)
    name === :subset_temperature_only_langevin && return _debug_subset_then_langevin(g; status = false, layer = false, kinetic = false, temperature = true, hamiltonian = false, magnetization = false, kwargs...)
    name === :subset_kinetic_only_langevin && return _debug_subset_then_langevin(g; status = false, layer = false, kinetic = true, temperature = false, hamiltonian = false, magnetization = false, kwargs...)
    name === :subset_layer_temperature_langevin && return _debug_subset_then_langevin(g; status = false, layer = true, kinetic = true, temperature = true, hamiltonian = false, magnetization = false, kwargs...)
    name === :subset_layer_temperature_no_kinetic_langevin && return _debug_subset_then_langevin(g; status = false, layer = true, kinetic = false, temperature = true, hamiltonian = false, magnetization = false, kwargs...)
    name === :subset_layer_kinetic_langevin && return _debug_subset_then_langevin(g; status = false, layer = true, kinetic = true, temperature = false, hamiltonian = false, magnetization = false, kwargs...)
    name === :subset_layer_hamiltonian_langevin && return _debug_subset_then_langevin(g; status = false, layer = false, kinetic = false, temperature = false, hamiltonian = true, magnetization = false, kwargs...)
    name === :subset_layer_magnetization_langevin && return _debug_subset_then_langevin(g; status = false, layer = true, kinetic = false, temperature = false, hamiltonian = false, magnetization = true, kwargs...)
    name === :subset_status_layer_temperature_langevin && return _debug_subset_then_langevin(g; status = true, layer = true, kinetic = true, temperature = true, hamiltonian = false, magnetization = false, kwargs...)
    name === :subset_full_no_status_langevin && return _debug_subset_then_langevin(g; status = false, layer = false, kinetic = true, temperature = true, hamiltonian = true, magnetization = true, kwargs...)
    name === :subset_full_no_temperature_langevin && return _debug_subset_then_langevin(g; status = true, layer = false, kinetic = true, temperature = false, hamiltonian = true, magnetization = true, kwargs...)
    name === :subset_full_no_hamiltonian_langevin && return _debug_subset_then_langevin(g; status = true, layer = true, kinetic = true, temperature = true, hamiltonian = false, magnetization = true, kwargs...)
    name === :subset_full_no_magnetization_langevin && return _debug_subset_then_langevin(g; status = true, layer = false, kinetic = true, temperature = true, hamiltonian = true, magnetization = false, kwargs...)

    throw(ArgumentError("Unknown close-debug window $name. Use close_debug_window_names()."))
end

"""
    open_close_debug_window(name::Symbol; kwargs...)

Open a graph-less close-debug window. This method is only valid for debug
windows whose `needs_graph` description is `false`.
"""
open_close_debug_window(name::Symbol; kwargs...) =
    open_close_debug_window(nothing, name; kwargs...)

"""
    _debug_title(name; title = nothing)

Return the native window title for a debug window, unless an explicit title was
provided by the caller.
"""
function _debug_title(name; title = nothing)
    isnothing(title) || return title
    return "Close Debug: $(replace(String(name), "_" => " "))"
end

"""
    _debug_raw_glmakie_label(; kwargs...)

Open a plain GLMakie `Screen` with a single label. Tests whether GLMakie alone,
without `WindowHost`, package timers, panels, graph ownership, or process
cleanup, can reproduce the close freeze.
"""
function _debug_raw_glmakie_label(; title = _debug_title(:raw_glmakie_label), size = (820, 520), focus = true)
    fig = Figure(; size)
    Label(fig[1, 1], "raw GLMakie label\nno WindowHost\nno package timers", fontsize = 24)
    screen = GLMakie.Screen(; focus_on_show = focus)
    display(screen, fig)
    GLFW.SetWindowTitle(to_native(screen), title)
    _focus_native_window!(screen)
    return (; kind = :raw_glmakie_label, figure = fig, screen)
end

"""
    _debug_host(; kwargs...) -> WindowHost

Construct and display a `WindowHost` with configurable timer startup. This is
the base facility used by most close-debug windows.
"""
function _debug_host(; title, size = (820, 520), fps = 30, polling_rate = 10, start_timers = true, start_poll_timer = true, focus = true)
    fig = Figure(; size)
    host = WindowHost(fig; screen = nothing, fps, polling_rate, open = Observable(true), start_timers = false)
    host[:title] = title
    _debug_display_host!(host, title; start_timers, start_poll_timer, focus)
    return host
end

"""
    _debug_display_host!(host, title; kwargs...) -> WindowHost

Attach a GLMakie screen to a prebuilt debug host, install the close and Cmd+W
observers, and optionally start frame/poll timers. Tests the host display path
separately from panel construction.
"""
function _debug_display_host!(host::WindowHost, title; start_timers = true, start_poll_timer = true, focus = true)
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
    if start_timers
        _start_host_timers!(host)
        if !start_poll_timer
            close(host.poll_timer)
            host.poll_timer = nothing
        end
    end
    return host
end

"""
    _debug_host_no_timers(; kwargs...) -> WindowHost

Open a `WindowHost` with close observers but no frame or poll timers. Tests
whether host close wiring alone is enough to trigger the freeze.
"""
function _debug_host_no_timers(; title = _debug_title(:host_no_timers), kwargs...)
    host = _debug_host(; title, start_timers = false, kwargs...)
    Label(host.figure[1, 1], "WindowHost\nclose observer\nno frame/poll timers", fontsize = 24)
    return host
end

"""
    _debug_host_empty(; kwargs...) -> WindowHost

Open a normal `WindowHost` with active frame and poll timers but no panels.
Tests whether package timers alone can reproduce the close freeze.
"""
function _debug_host_empty(; title = _debug_title(:host_empty), kwargs...)
    host = _debug_host(; title, kwargs...)
    Label(host.figure[1, 1], "WindowHost\nframe/poll timers active\nno panels", fontsize = 24)
    return host
end

"""
    _debug_host_frame_label(; kwargs...) -> WindowHost

Open a host where the frame timer mutates a label observable. Tests frame timer
notification without graph state or panels.
"""
function _debug_host_frame_label(; title = _debug_title(:host_frame_label), kwargs...)
    host = _debug_host(; title, kwargs...)
    ticks = Observable(0)
    Label(host.figure[1, 1], lift(i -> "frame callback ticks: $i", ticks), fontsize = 24)
    register_frame!(host) do _
        ticks[] += 1
        return nothing
    end
    return host
end

"""
    _debug_host_polled_label(; kwargs...) -> WindowHost

Open a host where the polling timer updates a `PolledObservable` shown in a
label. Tests host polling without graph state or panels.
"""
function _debug_host_polled_label(; title = _debug_title(:host_polled_label), kwargs...)
    host = _debug_host(; title, kwargs...)
    counter = Ref(0)
    po = register_polled!(host, PolledObservable(0, _ -> (counter[] += 1)))
    Label(host.figure[1, 1], lift(i -> "polled value: $i", po), fontsize = 24)
    return host
end

"""
    DebugStaticLayerPanel(graph, layer_idx)

Debug panel that plots one layer snapshot once and registers no frame callback.
Tests static graph-state plotting without live `notify` calls.
"""
struct DebugStaticLayerPanel <: AbstractPanel
    graph::Any
    layer_idx::Any
end

"""
    DebugCopyLayerPanel(graph, layer_idx)

Debug panel that displays a copied layer buffer and refreshes that owned buffer
on every frame before notifying Makie. Tests whether separating Makie's plot
data from the graph's live state memory changes close behavior.
"""
struct DebugCopyLayerPanel <: AbstractPanel
    graph::Any
    layer_idx::Any
end

"""
    DebugEmptyOnCloseLayerPanel(graph, layer_idx)

Debug panel that behaves like a live layer view, but registers a shutdown
resource that replaces the plot observable value with an empty array before
GLMakie screen close. Tests whether the close freeze depends on plot
observables still referencing graph state during teardown.
"""
struct DebugEmptyOnCloseLayerPanel <: AbstractPanel
    graph::Any
    layer_idx::Any
end

"""
    DebugSimulationSubsetPanel(graph; kwargs...)

Debug composite that mounts selected pieces of the simulation UI. Tests which
combination of subpanels is required to reproduce the close freeze.
"""
struct DebugSimulationSubsetPanel <: AbstractPanel
    graph::Any
    status::Bool
    layer::Bool
    kinetic::Bool
    temperature::Bool
    hamiltonian::Bool
    magnetization::Bool
end

"""
    DebugSimulationSubsetPanel(graph; status, layer, kinetic, temperature, hamiltonian, magnetization)

Normalize the subset flags to `Bool` so debug scenarios are stable across
caller-provided truthy values.
"""
function DebugSimulationSubsetPanel(
    graph;
    status = true,
    layer = true,
    kinetic = true,
    temperature = true,
    hamiltonian = true,
    magnetization = true,
)
    return DebugSimulationSubsetPanel(graph, Bool(status), Bool(layer), Bool(kinetic), Bool(temperature), Bool(hamiltonian), Bool(magnetization))
end

"""
    mount!(panel::DebugSimulationSubsetPanel, host, cell; kwargs...)

Mount only the selected simulation subpanels into a simplified layout. Used by
the suspect sequence to isolate panel interactions.
"""
function mount!(panel::DebugSimulationSubsetPanel, host::WindowHost, cell; kwargs...)
    grid = GridLayout(cell, alignmode = Outside(24), default_rowgap = 12)
    handle = PanelHandle(panel, host, grid)
    handle[:graph] = panel.graph
    handle[:layer_idx] = Observable(1)
    _register_graph_close!(handle, panel.graph)

    if panel.status
        status_handle = panel!(handle, :status, StatusPanel(panel.graph, handle[:layer_idx]), (1, 1))
        if _has_layer_selector(panel.graph)
            panel!(status_handle, :layer_selector, LayerSelectorPanel(panel.graph, handle[:layer_idx]), status_handle[:selector_slot])
        end
    else
        Label(grid[1, 1], "no status", fontsize = 12, tellheight = false)
        rowsize!(grid, 1, 32)
    end

    midgrid = GridLayout(grid[2, 1], default_colgap = 16)
    if panel.hamiltonian
        panel!(
            handle,
            :hamiltonian_parameters,
            HamiltonianParameterPanel(panel.graph, handle[:layer_idx], midgrid[1, 2]),
            midgrid[1, 1],
        )
        colsize!(midgrid, 1, 260)
        colsize!(midgrid, 2, Auto(false))
    elseif panel.layer
        panel!(handle, :layer_view, LayerViewPanel(panel.graph, handle[:layer_idx]), midgrid[1, 1])
        colsize!(midgrid, 1, Auto(false))
    else
        Label(midgrid[1, 1], "no central layer/hamiltonian display", fontsize = 18)
        colsize!(midgrid, 1, Auto(false))
    end

    if panel.kinetic || panel.temperature
        rightgrid = GridLayout(midgrid[1, 3], tellwidth = false, default_rowgap = 8)
        if panel.kinetic
            panel!(handle, :kinetic_time, KineticTimePanel(panel.graph), rightgrid[1, 1])
        end
        if panel.temperature
            panel!(handle, :temperature, TemperaturePanel(panel.graph), rightgrid[panel.kinetic ? 2 : 1, 1])
        end
        colsize!(midgrid, 3, 140)
    end

    if panel.magnetization
        panel!(handle, :magnetization, MagnetizationPanel(panel.graph, handle[:layer_idx]), (3, 1))
        rowsize!(grid, 3, 140)
    end
    return handle
end

"""
    mount!(panel::DebugStaticLayerPanel, host, cell; kwargs...)

Mount a graph layer as a one-time static plot. No graph close registration or
frame callback is installed.
"""
function mount!(panel::DebugStaticLayerPanel, host::WindowHost, cell; kwargs...)
    grid = GridLayout(cell)
    handle = PanelHandle(panel, host, grid)
    handle[:grid] = grid
    _with_layer(panel.graph, panel.layer_idx) do layer
        _debug_draw_static_layer!(handle, grid, layer)
    end
    return handle
end

"""
    _debug_draw_static_layer!(handle, grid, layer)

Draw a static layer snapshot for the static-layer debug window.
"""
function _debug_draw_static_layer!(handle, grid, layer::AbstractIsingLayer{T,2}) where {T}
    ax = handle[:axis] = Axis(grid[1, 1], aspect = DataAspect())
    vals = copy(state(layer))
    plot = image!(ax, vals, colormap = :thermal, fxaa = false, interpolate = false)
    _bind_layer_colorrange!(plot, Observable(vals), layer)
    reset_limits!(ax)
    return handle
end

function _debug_draw_static_layer!(handle, grid, layer::AbstractIsingLayer{T,3}) where {T}
    ax = handle[:axis] = Axis3(grid[1, 1])
    xs, ys, zs = _coordinates_3d!(handle, size(layer))
    vals = _cast_layer_state_vector(layer)
    plot = meshscatter!(ax, xs, ys, zs, markersize = 0.3, color = vals, colormap = :thermal)
    _bind_layer_colorrange!(plot, Observable(vals), layer)
    return handle
end

"""
    mount!(panel::DebugEmptyOnCloseLayerPanel, host, cell; kwargs...)

Mount a live layer display whose observable is emptied during host runtime
shutdown. The frame callback still notifies while the window is open.
"""
function mount!(panel::DebugEmptyOnCloseLayerPanel, host::WindowHost, cell; kwargs...)
    grid = GridLayout(cell)
    handle = PanelHandle(panel, host, grid)
    _register_graph_close!(handle, panel.graph)
    handle[:grid] = grid
    _redraw_debug_empty_on_close_layer!(handle)
    register!(handle, on(panel.layer_idx) do _
        _redraw_debug_empty_on_close_layer!(handle)
    end)
    register_frame!(handle) do _
        if haskey(handle, :img_obs)
            notify(handle[:img_obs])
        end
        return nothing
    end
    return handle
end

"""
    _redraw_debug_empty_on_close_layer!(handle)

Rebuild the empty-on-close layer plot when the selected layer changes.
"""
function _redraw_debug_empty_on_close_layer!(handle::PanelHandle)
    if haskey(handle, :axis)
        _remember_axis3_state!(handle, :axis3_state, handle[:axis])
        _delete_makie_object!(handle, handle[:axis])
    end

    panel = handle.panel::DebugEmptyOnCloseLayerPanel
    _with_layer(panel.graph, panel.layer_idx) do layer
        _draw_debug_empty_on_close_layer!(handle, handle[:grid], layer)
    end
    return handle
end

"""
    _draw_debug_empty_on_close_layer!(handle, grid, layer)

Draw the empty-on-close live layer display for 2D or 3D layers.
"""
function _draw_debug_empty_on_close_layer!(handle, grid, layer::AbstractIsingLayer{T,2}) where {T}
    ax = handle[:axis] = Axis(grid[1, 1], xrectzoom = false, yrectzoom = false, aspect = DataAspect(), tellheight = true)
    ax.yreversed = @load_preference("makie_y_flip", default = false)
    vals = _layer_state_view(layer)
    obs = handle[:img_obs] = hot_observable!(handle, vals)
    plot = handle[:plot] = image!(ax, obs, colormap = :thermal, fxaa = false, interpolate = false)
    _bind_layer_colorrange!(plot, obs, layer)
    reset_limits!(ax)
    return handle
end

function _draw_debug_empty_on_close_layer!(handle, grid, layer::AbstractIsingLayer{T,3}) where {T}
    ax = handle[:axis] = Axis3(grid[1, 1], tellheight = true)
    _restore_axis3_state!(ax, get(handle.data, :axis3_state, nothing))
    xs, ys, zs = _coordinates_3d!(handle, size(layer))
    vals = _layer_state_vector_view(layer)
    obs = handle[:img_obs] = hot_observable!(handle, vals)
    plot = handle[:plot] = meshscatter!(ax, xs, ys, zs, markersize = 0.3, color = obs, colormap = :thermal)
    _bind_layer_colorrange!(plot, obs, layer)
    return handle
end

"""
    mount!(panel::DebugCopyLayerPanel, host, cell; kwargs...)

Mount a copied live layer display. The frame callback refreshes the owned
display buffer from graph state and then notifies Makie.
"""
function mount!(panel::DebugCopyLayerPanel, host::WindowHost, cell; kwargs...)
    grid = GridLayout(cell)
    handle = PanelHandle(panel, host, grid)
    _register_graph_close!(handle, panel.graph)
    handle[:grid] = grid
    _redraw_debug_copy_layer!(handle)
    register!(handle, on(panel.layer_idx) do _
        _redraw_debug_copy_layer!(handle)
    end)
    register_frame!(handle) do _
        _refresh_debug_copy_layer!(handle)
        return nothing
    end
    return handle
end

"""
    _redraw_debug_copy_layer!(handle)

Rebuild the copied layer plot when the selected layer changes.
"""
function _redraw_debug_copy_layer!(handle::PanelHandle)
    if haskey(handle, :axis)
        _remember_axis3_state!(handle, :axis3_state, handle[:axis])
        _delete_makie_object!(handle, handle[:axis])
    end

    panel = handle.panel::DebugCopyLayerPanel
    _with_layer(panel.graph, panel.layer_idx) do layer
        _draw_debug_copy_layer!(handle, handle[:grid], layer)
    end
    return handle
end

"""
    _draw_debug_copy_layer!(handle, grid, layer)

Draw the copied live layer display for 2D or 3D layers.
"""
function _draw_debug_copy_layer!(handle, grid, layer::AbstractIsingLayer{T,2}) where {T}
    ax = handle[:axis] = Axis(grid[1, 1], xrectzoom = false, yrectzoom = false, aspect = DataAspect(), tellheight = true)
    ax.yreversed = @load_preference("makie_y_flip", default = false)
    vals = copy(state(layer))
    obs = handle[:img_obs] = Observable(vals)
    handle[:copy_src_layer] = layer
    plot = handle[:plot] = image!(ax, obs, colormap = :thermal, fxaa = false, interpolate = false)
    _bind_layer_colorrange!(plot, obs, layer)
    reset_limits!(ax)
    return handle
end

function _draw_debug_copy_layer!(handle, grid, layer::AbstractIsingLayer{T,3}) where {T}
    ax = handle[:axis] = Axis3(grid[1, 1], tellheight = true)
    _restore_axis3_state!(ax, get(handle.data, :axis3_state, nothing))
    xs, ys, zs = _coordinates_3d!(handle, size(layer))
    vals = Float64.(vec(state(layer)))
    obs = handle[:img_obs] = Observable(vals)
    handle[:copy_src_layer] = layer
    plot = handle[:plot] = meshscatter!(ax, xs, ys, zs, markersize = 0.3, color = obs, colormap = :thermal)
    _bind_layer_colorrange!(plot, obs, layer)
    return handle
end

"""
    _refresh_debug_copy_layer!(handle)

Copy current graph state into the owned display buffer and notify Makie.
"""
function _refresh_debug_copy_layer!(handle)
    haskey(handle, :img_obs) || return nothing
    haskey(handle, :copy_src_layer) || return nothing
    _copy_layer_state_to_display!(handle[:img_obs][], handle[:copy_src_layer])
    notify(handle[:img_obs])
    return nothing
end

"""
    _copy_layer_state_to_display!(dest, layer)

Copy layer state into an existing 2D or 3D display buffer without changing the
observable object identity.
"""
function _copy_layer_state_to_display!(dest::AbstractMatrix, layer)
    copyto!(dest, state(layer))
    return dest
end

function _copy_layer_state_to_display!(dest::AbstractVector, layer)
    src = vec(state(layer))
    @inbounds for i in eachindex(dest, src)
        dest[i] = src[i]
    end
    return dest
end

"""
    _debug_static_layer(g; kwargs...) -> WindowHost

Open the static-layer debug window: one graph layer plot, no live frame notify.
"""
function _debug_static_layer(g; title = _debug_title(:static_layer), kwargs...)
    host = _debug_host(; title, kwargs...)
    layer_idx = Observable(1)
    host[:layer_idx] = layer_idx
    host[:static_layer] = panel!(host, :static_layer, DebugStaticLayerPanel(g, layer_idx), (1, 1))
    return host
end

"""
    _debug_layer_view(g; kwargs...) -> WindowHost

Open the production `LayerViewPanel` by itself. Tests live frame `notify` on
the normal view-backed layer observable.
"""
function _debug_layer_view(g; title = _debug_title(:layer_view), kwargs...)
    host = _debug_host(; title, kwargs...)
    layer_idx = Observable(1)
    host[:layer_idx] = layer_idx
    host[:layer_view] = panel!(host, :layer_view, LayerViewPanel(g, layer_idx), (1, 1))
    return host
end

"""
    _debug_all_layers_view(g; kwargs...) -> WindowHost

Open the production `AllLayersViewPanel` by itself. Tests the MNIST-like path
where several layer plots are notified from one frame callback.
"""
function _debug_all_layers_view(g; title = _debug_title(:all_layers_view), size = (1200, 820), kwargs...)
    _debug_ensure_layer_coords!(g)
    host = _debug_host(; title, size, kwargs...)
    host[:all_layers_view] = panel!(host, :all_layers_view, AllLayersViewPanel(g), (1, 1))
    return host
end

"""
    _debug_ensure_layer_coords!(g)

Assign simple non-overlapping xy coordinates to graph layers that do not
already have coordinates. This keeps the all-layers close-debug window usable
with ordinary toy graphs.
"""
function _debug_ensure_layer_coords!(g::T) where {T}
    x = 0
    for layer in layers(g)
        has_coords = try
            !isnothing(coords(layer))
        catch
            false
        end
        if !has_coords
            _debug_package_module().setcoords!(layer; x, y = 0, z = 0)
        end
        x += maximum(size(layer)) + 2
    end
    return g
end

"""
    _debug_copy_layer_view(g; kwargs...) -> WindowHost

Open the copied live layer panel by itself. Tests whether an owned display
buffer changes close behavior compared with `LayerViewPanel`.
"""
function _debug_copy_layer_view(g; title = _debug_title(:layer_copy), kwargs...)
    host = _debug_host(; title, kwargs...)
    layer_idx = Observable(1)
    host[:layer_idx] = layer_idx
    host[:layer_copy] = panel!(host, :layer_copy, DebugCopyLayerPanel(g, layer_idx), (1, 1))
    return host
end

"""
    _debug_empty_on_close_layer_view(g; kwargs...) -> WindowHost

Open a live layer panel that empties its plot observable during shutdown before
GLMakie screen close. Tests whether close freezes are tied to live observable
memory still pointing into graph state.
"""
function _debug_empty_on_close_layer_view(g; title = _debug_title(:layer_empty_on_close), kwargs...)
    host = _debug_host(; title, kwargs...)
    layer_idx = Observable(1)
    host[:layer_idx] = layer_idx
    host[:layer_empty_on_close] = panel!(host, :layer_empty_on_close, DebugEmptyOnCloseLayerPanel(g, layer_idx), (1, 1))
    return host
end

"""
    _debug_temperature(g; kwargs...) -> WindowHost

Open the production `TemperaturePanel` by itself. Tests slider interaction plus
graph-backed temperature polling.
"""
function _debug_temperature(g; title = _debug_title(:temperature), kwargs...)
    host = _debug_host(; title, size = (420, 720), kwargs...)
    host[:temperature] = panel!(host, :temperature, TemperaturePanel(g), (1, 1))
    return host
end

"""
    _debug_status(g; kwargs...) -> WindowHost

Open the production `StatusPanel` by itself. Tests step counter polling and the
pause button without the layer view.
"""
function _debug_status(g; title = _debug_title(:status), kwargs...)
    host = _debug_host(; title, size = (1000, 300), kwargs...)
    layer_idx = Observable(1)
    host[:layer_idx] = layer_idx
    host[:status] = panel!(host, :status, StatusPanel(g, layer_idx), (1, 1))
    return host
end

"""
    _debug_magnetization(g; kwargs...) -> WindowHost

Open the production `MagnetizationPanel` by itself. Tests graph-state polling
without image or mesh rendering.
"""
function _debug_magnetization(g; title = _debug_title(:magnetization), kwargs...)
    host = _debug_host(; title, size = (760, 360), kwargs...)
    layer_idx = Observable(1)
    host[:layer_idx] = layer_idx
    host[:magnetization] = panel!(host, :magnetization, MagnetizationPanel(g, layer_idx), (1, 1))
    return host
end

"""
    _debug_hamiltonian(g; kwargs...) -> WindowHost

Open the production `HamiltonianParameterPanel` by itself. Tests selector
buttons and midpanel Hamiltonian visualizations without the rest of the UI.
"""
function _debug_hamiltonian(g; title = _debug_title(:hamiltonian), kwargs...)
    host = _debug_host(; title, size = (1100, 720), kwargs...)
    layer_idx = Observable(1)
    host[:layer_idx] = layer_idx
    grid = GridLayout(host.figure[1, 1], default_colgap = 16)
    host[:midpanel] = grid[1, 2]
    host[:hamiltonian] = panel!(
        host,
        :hamiltonian,
        HamiltonianParameterPanel(g, layer_idx, host[:midpanel]),
        grid[1, 1],
    )
    colsize!(grid, 1, 260)
    colsize!(grid, 2, Auto(false))
    return host
end

"""
    _debug_simulation(g; hide_left_buttons, kwargs...) -> WindowHost

Open the production `SimulationPanel` in a debug host. Tests the full composite
or the same composite with Hamiltonian selector buttons hidden.
"""
function _debug_simulation(g; hide_left_buttons, title = _debug_title(hide_left_buttons ? :simulation_hidden_left : :simulation_full), kwargs...)
    host = _debug_host(; title, size = (1500, 1000), kwargs...)
    host[:simulation] = panel!(host, :simulation, SimulationPanel(g; hide_left_buttons), (1, 1))
    return host
end

"""
    _debug_simulation_subset(g; kwargs...) -> WindowHost

Open a simplified simulation-like layout with a chosen subset of subpanels.
Used to isolate the smallest panel combination that reproduces the close issue.
"""
function _debug_simulation_subset(
    g;
    status = true,
    layer = true,
    kinetic = true,
    temperature = true,
    hamiltonian = true,
    magnetization = true,
    title = _debug_title(:simulation_subset),
    kwargs...,
)
    host = _debug_host(; title, size = (1500, 1000), kwargs...)
    host[:simulation_subset] = panel!(
        host,
        :simulation_subset,
        DebugSimulationSubsetPanel(
            g;
            status,
            layer,
            kinetic,
            temperature,
            hamiltonian,
            magnetization,
        ),
        (1, 1),
    )
    return host
end

"""
    _debug_public_interface(g; kwargs...) -> WindowHost

Open the exact public `interface(g)` entrypoint for comparison with debug-only
hosts.
"""
function _debug_public_interface(g; title = _debug_title(:public_interface), kwargs...)
    return interface(g; title, kwargs...)
end

"""
    _debug_package_module()

Return the parent package module so the debug code can access public graph and
process constructors without importing them into `Windows`.
"""
_debug_package_module() = parentmodule(@__MODULE__)

"""
    _debug_langevin_algorithm()

Return the `LocalLangevin(adjusted = false)` algorithm used by the running
process close reproducers.
"""
function _debug_langevin_algorithm()
    return _debug_package_module().LocalLangevin(adjusted = false)
end

"""
    _debug_start_langevin_process!(g; kwargs...)

Start a `LocalLangevin` process on `g` for debug windows.
"""
function _debug_start_langevin_process!(g; algorithm = _debug_langevin_algorithm(), kwargs...)
    return _debug_package_module().createProcess(g, algorithm; allow_multiple = true, kwargs...)
end

"""
    _debug_start_process!(g, algorithm; kwargs...)

Start either the provided debug process algorithm or the package default
process when `algorithm === nothing`.
"""
function _debug_start_process!(g, algorithm; kwargs...)
    if isnothing(algorithm)
        return _debug_package_module().createProcess(g; allow_multiple = true, kwargs...)
    else
        return _debug_package_module().createProcess(g, algorithm; allow_multiple = true, kwargs...)
    end
end

"""
    _debug_with_process(g, opener; process_algorithm = _debug_langevin_algorithm(), kwargs...)

Start a graph process before opening the window. Tests the "process already
running at display time" path. The default process algorithm is the debug
Langevin algorithm, but callers can pass `process_algorithm = nothing` to use
the package default process or pass a concrete algorithm to test another graph
mutation path.
"""
function _debug_with_process(g, opener; process_algorithm = _debug_langevin_algorithm(), process_kwargs = NamedTuple(), kwargs...)
    _debug_start_process!(g, process_algorithm; process_kwargs...)
    return opener(g; kwargs...)
end

"""
    _debug_open_then_process(g, opener, algorithm; process_algorithm = algorithm, kwargs...)

Open the window first, then start a process on the same graph. Tests the common
example path `host = interface(g); createProcess(g, ...)`. `algorithm` is the
scenario default; pass `process_algorithm` to override it for an otherwise
identical window.
"""
function _debug_open_then_process(g, opener, algorithm; process_algorithm = algorithm, process_kwargs = NamedTuple(), kwargs...)
    host = opener(g; kwargs...)
    host[:debug_process] = _debug_start_process!(g, process_algorithm; process_kwargs...)
    return host
end

"""
    _debug_subset_then_langevin(g; process_algorithm = _debug_langevin_algorithm(), kwargs...)

Open a selected simulation subset first, then start a graph process. The name is
kept for existing debug sequences, but the diagnostic should be interpreted as
"subset plus running graph process", not as a Langevin-specific test.
"""
function _debug_subset_then_langevin(
    g;
    status,
    layer,
    kinetic,
    temperature,
    hamiltonian,
    magnetization,
    title = _debug_title(:simulation_subset_langevin),
    process_algorithm = _debug_langevin_algorithm(),
    process_kwargs = NamedTuple(),
    kwargs...,
)
    opener = (gg; kwargs...) -> _debug_simulation_subset(
        gg;
        status,
        layer,
        kinetic,
        temperature,
        hamiltonian,
        magnetization,
        title,
        kwargs...,
    )
    return _debug_open_then_process(g, opener, process_algorithm; process_kwargs, kwargs...)
end

"""
    _debug_simulation_sync_close_graph(g; kwargs...) -> WindowHost

Open the full simulation panel and deliberately register a bad synchronous
`Processes.close(g)` close callback. This is a negative-control window.
"""
function _debug_simulation_sync_close_graph(g; title = _debug_title(:simulation_bad_sync_close_graph), kwargs...)
    host = _debug_simulation(g; hide_left_buttons = false, title, kwargs...)
    onclose!(host) do _
        Processes.close(g)
    end
    return host
end

"""
    _debug_simulation_async_close_graph(g; kwargs...) -> WindowHost

Open the full simulation panel and register an async graph close callback.
Tests whether explicit example-level graph cleanup changes close behavior.
"""
function _debug_simulation_async_close_graph(g; title = _debug_title(:simulation_async_close_graph), kwargs...)
    host = _debug_simulation(g; hide_left_buttons = false, title, kwargs...)
    onclose!(host) do _
        @async Processes.close(g)
    end
    return host
end
