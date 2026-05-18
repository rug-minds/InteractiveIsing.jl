function _precompile_weight(; dr)
    return dr == 1 ? 1.0f0 : 0.0f0
end

function _precompile_windows_interface(g)
    host = Windows.WindowHost(Figure(); screen = nothing, fps = 30, polling_rate = 10, start_timers = false)
    handle = Windows.panel!(host, Windows.SimulationPanel(g), (1, 1))
    _precompile_simulation_runtime!(host, handle)
    close(host)
    return nothing
end

function _precompile_simulation_runtime!(host, handle)
    Windows._tick!(host)
    Windows._poll!(host)

    if haskey(handle.children, :hamiltonian_parameters)
        parameter_panel = handle.children[:hamiltonian_parameters]
        entries = parameter_panel[:entries]
        for idx in eachindex(entries)
            parameter_panel[:selected][] = idx
            Windows._draw_hamiltonian_entry!(parameter_panel)
            Windows._refresh_hamiltonian_display!(parameter_panel)
        end
    end

    if haskey(handle.children, :status) &&
            haskey(handle.children[:status].children, :layer_selector)
        selector = handle.children[:status].children[:layer_selector]
        selector.panel.layer_idx[] = min(2, length(layers(handle.panel.graph)))
        selector.panel.layer_idx[] = 1
    end

    Windows.tofigure(handle; size = (600, 420))
    Windows.tofigure(host; size = (600, 420))
    return nothing
end

function _precompile_context_lines()
    x = Float32[1, 2, 3]
    y = Float32[1, 4, 9]
    host_lines = Windows.WindowHost(Figure(); screen = nothing, fps = 30, polling_rate = 10, start_timers = false)
    handle = Windows.panel!(host_lines, Windows.InteractiveLinesPanel(x, y; update_rate = 10), (1, 1))
    Windows._poll!(host_lines)
    Windows.tofigure(handle; size = (420, 320))
    close(host_lines)

    ctx = Processes.ProcessContext(
        (;
            demo = Processes.SubContext(:demo, (; x = Float32[1, 2, 3], y = Float32[1, 4, 9])),
            globals = (;),
        ),
        Processes.NameSpaceRegistry(),
    )
    host = Windows.WindowHost(Figure(); screen = nothing, fps = 30, polling_rate = 10, start_timers = false)
    Windows.panel!(
        host,
        Windows.ContextLinesPanel(
            ctx,
            :demo => :x,
            :demo => :y;
            xlabel = "x",
            ylabel = "y",
            title = "trace",
            update_rate = 10,
        ),
        (1, 1),
    )
    Windows._poll!(host)
    close(host)
    return nothing
end

function _precompile_connections(g)
    host = Windows.WindowHost(Figure(); screen = nothing, fps = 30, polling_rate = 10, start_timers = false)
    handle = Windows.panel!(host, Windows.ConnectionsPanel(g; max_edges = 8), (1, 1))
    Windows.tofigure(handle; size = (420, 320))
    close(host)
    return nothing
end

function _precompile_layer_view(g)
    host = Windows.WindowHost(Figure(); screen = nothing, fps = 30, polling_rate = 10, start_timers = false)
    layer_idx = Observable(1)
    handle = Windows.panel!(host, Windows.LayerViewPanel(g, layer_idx), (1, 1))
    Windows._tick!(host)
    Windows.tofigure(handle; size = (420, 320))
    close(host)
    return nothing
end

function _precompile_all_layers_view(g)
    host = Windows.WindowHost(Figure(); screen = nothing, fps = 30, polling_rate = 10, start_timers = false)
    handle = Windows.panel!(host, Windows.AllLayersViewPanel(g; labels = true), (1, 1))
    Windows._tick!(host)
    Windows.tofigure(handle; size = (420, 320))
    close(host)
    return nothing
end

function _precompile_gl_window(g)
    get(ENV, "INTERACTIVEISING_PRECOMPILE_GL_WINDOWS", "1") == "0" && return nothing

    host = nothing
    screen = nothing
    try
        host = Windows.WindowHost(
            Figure(; size = (420, 320));
            screen = nothing,
            fps = 30,
            polling_rate = 10,
            start_timers = false,
        )
        handle = Windows.panel!(host, Windows.SimulationPanel(g), (1, 1))
        _precompile_simulation_runtime!(host, handle)

        screen = display(GLMakie.Screen(visible = false), host.figure)
        GLMakie.render_frame(screen; resize_buffers = false)
        GLMakie.Makie.colorbuffer(screen)
    catch err
        @debug "Skipping GLMakie hidden-window precompile warmup" exception = (err, catch_backtrace())
    finally
        isnothing(host) || close(host)
        if !isnothing(screen)
            try
                GLMakie.destroy!(screen)
            catch err
                @debug "Could not destroy GLMakie precompile screen" exception = (err, catch_backtrace())
            end
        end
    end
    return nothing
end

function _precompile_processes(g2, g3)
    p1 = createProcess(g2, Metropolis(); lifetime = 1)
    wait(p1)
    Processes.close(g2)

    p2 = createProcess(g3, LocalLangevin(stepsize = 0.05f0, adjusted = false); lifetime = 1)
    wait(p2)
    Processes.close(g3)
    return nothing
end

function _precompile_windows_entrypoints(g)
    precompile(Tuple{typeof(interface), typeof(g)})
    precompile(Tuple{typeof(Windows.interface), typeof(g)})
    precompile(Tuple{typeof(Windows.new_interface), typeof(g)})
    precompile(Tuple{typeof(Windows._display_host!), Windows.WindowHost, String})
    precompile(Tuple{typeof(Windows._start_host_timers!), Windows.WindowHost})
    precompile(Tuple{typeof(Windows.window)})
    return nothing
end

@setup_workload begin
    @compile_workload begin
        g2 = IsingGraph(4, 4, Continuous(), StateSet(-1.0f0, 1.0f0); precision = Float32)
        g2_layers = IsingGraph(
            Layer(4, 4, Continuous(), StateSet(-1.0f0, 1.0f0)),
            Layer(4, 4, Continuous(), StateSet(-1.0f0, 1.0f0)),
            Ising(b = Float32[0.1 for _ in 1:32]) + Quartic();
            precision = Float32,
        )
        g2_positioned = IsingGraph(
            Layer(4, 4, Continuous(), StateSet(-1.0f0, 1.0f0), Coords(y = 0, x = 0, z = 0)),
            Layer(4, 4, Continuous(), StateSet(-1.0f0, 1.0f0), Coords(y = 0, x = 4, z = 0)),
            Ising(b = Float32[0.1 for _ in 1:32]) + Quartic();
            precision = Float32,
        )
        wg3 = @WG _precompile_weight NN = 1
        g3 = IsingGraph(
            4, 4, 3,
            Continuous(),
            wg3,
            LatticeConstants(1.0f0, 1.0f0, 1.0f0),
            StateSet(-1.5f0, 1.5f0),
            Ising(c = ConstVal(0.0f0), b = 0) + CoulombHamiltonian(recalc = 1);
            periodic = (:x, :y),
        )

        _precompile_windows_entrypoints(g2)
        _precompile_windows_entrypoints(g2_layers)
        _precompile_windows_entrypoints(g3)
        _precompile_windows_interface(g2)
        _precompile_windows_interface(g2_layers)
        _precompile_windows_interface(g3)
        _precompile_context_lines()
        _precompile_connections(g2)
        _precompile_connections(g3)
        _precompile_layer_view(g2)
        _precompile_layer_view(g3)
        _precompile_all_layers_view(g2_positioned)
        _precompile_gl_window(g2)
        _precompile_gl_window(g2_layers)
        _precompile_gl_window(g3)
        _precompile_processes(g2, g3)
    end
end
