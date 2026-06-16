using Test
using GLMakie
using InteractiveIsing
using InteractiveIsing.Windows
using Unitful

struct TestResource
    closed::Base.RefValue{Bool}
end

Base.close(resource::TestResource) = (resource.closed[] = true)

struct TestPanel <: Windows.AbstractPanel end

function Windows.mount!(panel::TestPanel, host::Windows.WindowHost, cell; kwargs...)
    handle = Windows.PanelHandle(panel, host, cell)
    handle[:resource] = Windows.register!(handle, TestResource(Ref(false)))
    return handle
end

function window_test_weights(; dr)
    return dr == 1 ? 1.0f0 : 0.0f0
end

@testset "Windows host lifecycle" begin
    host = Windows.WindowHost(Figure(); screen = nothing, fps = 30, polling_rate = 10)
    handle = Windows.panel!(host, :test, TestPanel(), (1, 1))
    resource = handle[:resource]

    @test !resource.closed[]
    @test host.children[:test] === handle
    @test handle.slot == (1, 1)
    @test sprint(show, host) == "WindowHost(\"untitled\", panels=1, resources=0, close_callbacks=0, fps=30.0, polling_rate=10.0, closed=false)"
    @test sprint(show, handle) == "PanelHandle(TestPanel, slot=(1, 1), children=0, resources=1, close_callbacks=0, closed=false)"
    @test sprint(show, TestPanel()) == "TestPanel(...)"
    @test !occursin("Figure", sprint(show, MIME("text/plain"), host))
    pause!(host)
    @test host.paused[]
    @test !StatefulAlgorithms.ispaused(host.frame_timer)
    @test !StatefulAlgorithms.ispaused(host.poll_timer)
    resume!(host)
    @test !host.paused[]
    close(host)
    @test !resource.closed[]
    @test host.closed

    close(host)
    @test !resource.closed[]

    direct_handle_host = Windows.WindowHost(Figure(); screen = nothing, fps = 30, polling_rate = 10)
    direct_handle = Windows.panel!(direct_handle_host, TestPanel(), (1, 1))
    direct_resource = direct_handle[:resource]
    close(direct_handle)
    @test direct_resource.closed[]

    callback_host = Windows.WindowHost(Figure(); screen = nothing, fps = 30, polling_rate = 10)
    callback_done = Channel{Symbol}(1)
    callback_owner = Ref{Any}(nothing)
    Windows.onclose!(callback_host) do owner
        callback_owner[] = owner
        put!(callback_done, :host)
    end
    callback_handle = Windows.panel!(callback_host, TestPanel(), (1, 1))
    panel_done = Channel{Symbol}(1)
    panel_owner = Ref{Any}(nothing)
    Windows.onclose!(callback_handle) do owner
        panel_owner[] = owner
        put!(panel_done, :panel)
    end
    close(callback_host)
    @test take!(callback_done) == :host
    @test callback_owner[] === callback_host
    @test !isready(panel_done)
    @test panel_owner[] === nothing

    native_host = Windows.WindowHost(Figure(); screen = nothing, fps = 30, polling_rate = 10)
    native_handle = Windows.panel!(native_host, :test, TestPanel(), (1, 1))
    native_resource = native_handle[:resource]
    native_host.open[] = false
    Windows._tick!(native_host)
    for _ in 1:20
        native_handle.closed && break
        sleep(0.01)
    end
    @test native_host.closed
    @test native_handle.closed
    @test !native_resource.closed[]
end

@testset "PolledObservable setter and polling" begin
    backing = Ref(1)
    po = InteractiveIsing.PolledObservable(backing[], _ -> backing[]; setter = x -> (backing[] = x))

    po[] = 2
    @test backing[] == 2
    @test po[] == 2

    backing[] = 3
    InteractiveIsing.poll!(po)
    @test po[] == 3
end

@testset "TemperaturePanel physical scale label" begin
    scales = PhysicalScales(energy = 1u"meV", temperature = u"meV")
    g = IsingGraph(
        2,
        2,
        Continuous();
        precision = Float64,
        physical_scales = scales,
        temperature = 3.0,
    )
    host = Windows.WindowHost(Figure(); screen = nothing, fps = 30, polling_rate = 10)
    handle = Windows.panel!(host, Windows.TemperaturePanel(g), (1, 1))

    @test handle[:label_text][] == "T: 34.81 K"

    temp!(g, 4.0)
    Windows._poll!(host)
    @test handle[:slider].value[] ≈ 4.0
    @test handle[:label_text][] == "T: 46.42 K"

    handle[:slider].value[] = 5.0
    @test temp(g) ≈ 5.0
    @test handle[:label_text][] == "T: 58.02 K"

    handle[:slider].value[] = 0.01
    @test temp(g) ≈ 0.01
    @test handle[:slider].value[] ≈ 0.01
    @test handle[:label_text][] == "T: 0.116 K"

    close(host)
end

@testset "Graph process resume includes finished processes" begin
    g = IsingGraph(2, 2, Continuous(), StateSet(-1.0f0, 1.0f0); precision = Float32)
    algorithm = Metropolis()
    process = StatefulAlgorithms.Process(algorithm, InteractiveIsing._mc_model_inits(algorithm, g)...; repeat = 1)
    push!(processes(g), process)

    run(process)
    wait(process)
    @test StatefulAlgorithms.isdone(process)
    @test Windows._graph_paused(g)

    previous_ticks = StatefulAlgorithms.getticks(process)
    Windows._resume_graph_processes!(g)
    wait(process)
    @test StatefulAlgorithms.getticks(process) > previous_ticks
    @test Windows._graph_paused(g)
end

@testset "PTimer close and wait" begin
    ticks = Channel{Symbol}(1)
    timer = PTimer(_ -> put!(ticks, :tick), 0; interval = 0.01)
    @test take!(ticks) == :tick
    close(timer)
    @test StatefulAlgorithms.ispaused(timer)
    @test wait(timer) === nothing
    StatefulAlgorithms.start(timer)
    @test !StatefulAlgorithms.ispaused(timer)
    @test take!(ticks) == :tick
    close(timer)
    @test wait(timer) === nothing
end

@testset "Hot observable cleanup" begin
    mat = rand(Float32, 3, 4)
    mat_view = view(mat, :, :)
    mat_obs = Observable{typeof(mat_view)}(mat_view)
    @test Windows.hot_observable_zero(mat_obs) isa typeof(mat_view)
    Windows.detach_hot_observable!(mat_obs)
    @test mat_obs[] isa typeof(mat_view)
    @test size(mat_obs[]) == (0, 0)
    @test parent(mat_obs[]) !== mat

    cube = rand(Float32, 3, 4, 5)
    cube_vec = vec(view(cube, :, :, :))
    vec_obs = Observable{typeof(cube_vec)}(cube_vec)
    @test Windows.hot_observable_zero(vec_obs) isa typeof(cube_vec)
    Windows.detach_hot_observable!(vec_obs)
    @test vec_obs[] isa typeof(cube_vec)
    @test length(vec_obs[]) == 0
    @test parent(parent(vec_obs[])) !== cube

    abstract_obs = Observable{AbstractVector{Float64}}([1.0, 2.0])
    Windows.detach_hot_observable!(abstract_obs)
    @test abstract_obs[] isa AbstractVector{Float64}
    @test length(abstract_obs[]) == 0

    host = Windows.WindowHost(Figure(); screen = nothing, fps = 30, polling_rate = 10)
    live_obs = Windows.hot_observable!(host, view(mat, :, :))
    close(host)
    @test live_obs[] isa typeof(mat_view)
    @test size(live_obs[]) == (0, 0)
    @test parent(live_obs[]) !== mat

    scheduled_host = Windows.WindowHost(Figure(); screen = nothing, fps = 30, polling_rate = 10, start_timers = false)
    scheduled_obs = Windows.hot_observable!(scheduled_host, view(mat, :, :))
    Windows._schedule_native_close!(scheduled_host)
    @test scheduled_obs[] isa typeof(mat_view)
    @test size(scheduled_obs[]) == (0, 0)
    @test parent(scheduled_obs[]) !== mat
end

@testset "InteractiveLinesPanel and ContextLinesPanel figure construction" begin
    ctx = InteractiveIsing.StatefulAlgorithms.ProcessContext(
        (;
            demo = InteractiveIsing.StatefulAlgorithms.SubContext(:demo, (; x = collect(1:5), y = collect(2:2:6))),
        ),
        InteractiveIsing.StatefulAlgorithms.NameSpaceRegistry(),
    )
    host = Windows.WindowHost(Figure(); screen = nothing, fps = 30, polling_rate = 10)
    handle = Windows.panel!(
        host,
        Windows.ContextLinesPanel(
            ctx,
            :demo => :x,
            InteractiveIsing.StatefulAlgorithms.Var(:demo, :y);
            xlabel = "x",
            ylabel = "y",
            title = "tracked",
            line_kwargs = (; color = :red),
            update_rate = 0,
        ),
        (1, 1),
    )

    @test handle.panel isa Windows.InteractiveLinesPanel
    @test String(handle[:axis].xlabel[]) == "x"
    @test String(handle[:axis].ylabel[]) == "y"
    @test String(handle[:axis].title[]) == "tracked"
    @test handle[:x_container] === ctx.demo.x
    @test handle[:y_container] === ctx.demo.y
    @test parent(handle[:x_obs][]) === ctx.demo.x
    @test parent(handle[:y_obs][]) === ctx.demo.y
    @test length(handle[:x_obs][]) == 3
    @test length(handle[:y_obs][]) == 3
    @test collect(handle[:x_obs][]) == [1, 2, 3]
    @test collect(handle[:y_obs][]) == [2, 4, 6]

    ctx.demo.x[1] = 10
    push!(ctx.demo.y, 8, 10)
    Windows._poll!(host)
    @test parent(handle[:x_obs][]) === ctx.demo.x
    @test parent(handle[:y_obs][]) === ctx.demo.y
    @test length(handle[:x_obs][]) == 5
    @test length(handle[:y_obs][]) == 5
    @test collect(handle[:x_obs][]) == [10, 2, 3, 4, 5]
    @test collect(handle[:y_obs][]) == [2, 4, 6, 8, 10]

    close(host)
    @test host.closed

    probe = Ref(:probe)
    probe_ctx = Dict(probe => (; xs = [1, 2], ys = [3, 4]))
    @test Windows._context_var_value(probe_ctx, probe => :xs) == [1, 2]

    xs = Float64[]
    ys = Float64[]
    mat = Ref([1.0 2.0; 3.0 4.0])
    owned_host = Windows.WindowHost(Figure(); screen = nothing, fps = 30, polling_rate = 10)
    owned_handle = Windows.panel!(
        owned_host,
        Windows.InteractiveLinesPanel(
            () -> begin
                push!(xs, length(xs) + 1)
                push!(ys, sum(abs2, mat[]))
                return xs, ys
            end;
            update_rate = 0,
        ),
        (1, 1),
    )

    @test parent(owned_handle[:x_obs][]) === xs
    @test parent(owned_handle[:y_obs][]) === ys
    @test collect(owned_handle[:x_obs][]) == [1.0]
    @test collect(owned_handle[:y_obs][]) == [30.0]
    mat[] .= 2.0
    Windows._poll!(owned_host)
    @test collect(owned_handle[:x_obs][]) == [1.0, 2.0]
    @test collect(owned_handle[:y_obs][]) == [30.0, 16.0]

    close(owned_host)
    @test owned_host.closed
end

@testset "ConnectionsPanel figure construction" begin
    wg = @WG window_test_weights NN = 1
    g = IsingGraph(
        Layer(2, Continuous()),
        wg,
        Layer(2, Continuous());
        precision = Float32,
    )
    host = Windows.WindowHost(Figure(); screen = nothing, fps = 30, polling_rate = 10)
    handle = Windows.panel!(host, Windows.ConnectionsPanel(g; max_edges = 1), (1, 1))

    @test haskey(handle, :axis)
    @test haskey(handle, :edge_plot)
    @test haskey(handle, :node_plot)
    @test handle[:edge_count] == 2
    @test handle[:visible_edge_count] == 1
    @test length(handle[:edge_points][]) == 9
    @test length(handle[:edge_colors][]) == 9
    @test length(handle[:edge_weights]) == 1
    @test Windows.hasaxis(handle)
    @test Windows.getaxis(handle) === handle[:axis]
    @test Windows.hasimage(handle)
    @test Windows.tofigure(handle) isa Figure
    export_path = tempname() * ".png"
    @test Windows.axis_to_png(export_path, handle) == export_path
    @test isfile(export_path)
    @test filesize(export_path) > 0
    rm(export_path; force = true)
    @test Windows.toimage(export_path, handle; px_per_unit = 1) == export_path
    @test isfile(export_path)
    @test filesize(export_path) > 0
    rm(export_path; force = true)

    close(host)
    @test host.closed
end

@testset "AllLayersViewPanel figure construction" begin
    g = IsingGraph(
        Layer(2, 3, Continuous(), StateSet(-1.0f0, 1.0f0), Coords(y = 0, x = 0, z = 0)),
        Layer(2, 3, Continuous(), StateSet(-1.0f0, 1.0f0), Coords(y = 1, x = 4, z = 0));
        precision = Float32,
    )
    host = Windows.WindowHost(Figure(); screen = nothing, fps = 30, polling_rate = 10)
    handle = Windows.panel!(host, Windows.AllLayersViewPanel(g; labels = false), (1, 1))

    @test Windows.hasaxis(handle)
    @test Windows.hasimage(handle)
    @test Windows.getaxis(handle) === handle[:axis]
    @test length(handle[:placements]) == 2
    @test length(handle[:plots]) == 2
    @test handle[:placements][1].x0 == 0.0f0
    @test handle[:placements][1].x1 == 3.0f0
    @test handle[:placements][1].y0 == 0.0f0
    @test handle[:placements][1].y1 == 2.0f0
    @test handle[:placements][2].x0 == 4.0f0
    @test handle[:placements][2].x1 == 7.0f0
    @test handle[:placements][2].y0 == 1.0f0
    @test handle[:placements][2].y1 == 3.0f0
    @test Windows.tofigure(handle) isa Figure

    state(g[1]) .= 0.5f0
    Windows._tick!(host)
    @test all(handle[:layer_observables][1][] .== 0.5f0)
    all_layers_obs = copy(handle[:layer_observables])

    duplicate = IsingGraph(
        Layer(2, 2, Continuous(), StateSet(-1.0f0, 1.0f0), Coords(y = 0, x = 0, z = 0)),
        Layer(2, 2, Continuous(), StateSet(-1.0f0, 1.0f0), Coords(y = 0, x = 0, z = 0));
        precision = Float32,
    )
    duplicate_host = Windows.WindowHost(Figure(); screen = nothing, fps = 30, polling_rate = 10)
    @test_throws ArgumentError Windows.panel!(
        duplicate_host,
        Windows.AllLayersViewPanel(duplicate),
        (1, 1),
    )
    close(duplicate_host)

    missing_coords = IsingGraph(
        Layer(2, 2, Continuous(), StateSet(-1.0f0, 1.0f0)),
        Layer(2, 2, Continuous(), StateSet(-1.0f0, 1.0f0), Coords(y = 0, x = 1, z = 0));
        precision = Float32,
    )
    missing_host = Windows.WindowHost(Figure(); screen = nothing, fps = 30, polling_rate = 10)
    @test_throws ArgumentError Windows.panel!(
        missing_host,
        Windows.AllLayersViewPanel(missing_coords),
        (1, 1),
    )
    close(missing_host)

    close(host)
    @test host.closed
    @test all(obs -> size(obs[]) == (0, 0), all_layers_obs)
end

@testset "SimulationPanel figure construction" begin
    g = IsingGraph(3, 3, Continuous(), StateSet(-1.0f0, 1.0f0); precision = Float32)
    host = Windows.WindowHost(Figure(); screen = nothing, fps = 30, polling_rate = 10)
    handle = Windows.panel!(host, Windows.SimulationPanel(g), (1, 1))

    @test handle[:graph] === g
    @test handle[:layer_idx][] == 1
    @test haskey(handle.children, :status)
    @test haskey(handle.children, :hamiltonian_parameters)
    @test haskey(handle.children, :temperature)
    @test !haskey(handle.children[:status].children, :layer_selector)
    @test haskey(handle.children[:status], :graph_paused)
    @test !handle.children[:status][:graph_paused][]
    parameter_panel = handle.children[:hamiltonian_parameters]
    entries = parameter_panel[:entries]
    labels = getfield.(entries, :term_label)
    @test !parameter_panel[:buttons_hidden]
    @test length(parameter_panel[:selector_buttons]) == length(entries)
    @test Windows.hasaxis(parameter_panel)
    @test Windows.getaxis(parameter_panel) === parameter_panel[:display_axis]
    @test Windows.hasimage(handle)
    @test Windows.hasimage(handle.children[:status])
    @test Windows.hasimage(handle.children[:temperature])
    @test Windows.hasimage(handle.children[:magnetization])
    @test Windows.tofigure(parameter_panel) isa Figure
    @test Windows.tofigure(handle) isa Figure
    @test Windows.tofigure(host) isa Figure
    export_path = tempname() * ".png"
    @test Windows.toimage(export_path, handle) == export_path
    @test isfile(export_path)
    @test filesize(export_path) > 0
    rm(export_path; force = true)
    @test Windows.toimage(export_path, host; px_per_unit = 1) == export_path
    @test isfile(export_path)
    @test filesize(export_path) > 0
    rm(export_path; force = true)
    @test Windows.fullimage(export_path, host; px_per_unit = 1) == export_path
    @test isfile(export_path)
    @test filesize(export_path) > 0
    rm(export_path; force = true)
    @test !isempty(entries)
    @test :state in getfield.(entries, :name)
    @test :lp in getfield.(entries, :name)
    @test :b in getfield.(entries, :name)
    @test :c ∉ getfield.(entries, :name)
    @test :J ∉ getfield.(entries, :name)
    @test any(occursin("Quadratic", label) for label in labels)
    @test any(occursin("MagField", label) for label in labels)
    @test all(!occursin("{", label) for label in labels)
    @test all(!occursin("PolynomialHamiltonian", label) for label in labels)
    graph_state_obs = parameter_panel[:display_obs]
    graph_state_type = typeof(graph_state_obs[])
    @test graph_state_obs isa Observable{graph_state_type}
    @test graph_state_type === Matrix{Float32}

    display_notifications = Ref(0)
    on(parameter_panel[:display_obs]) do _
        display_notifications[] += 1
    end
    Windows._tick!(host)
    @test display_notifications[] > 0
    state(g[1]) .= 0.25f0
    Windows._tick!(host)
    @test all(parameter_panel[:display_obs][] .== 0.25f0)

    temp!(g, 2.0f0)
    Windows._tick!(host)
    temperature_panel = handle.children[:temperature]
    Windows._poll!(host)
    @test temperature_panel[:slider].value[] ≈ 2.0f0

    algorithm = Metropolis()
    process_inputs = InteractiveIsing._mc_model_inits(algorithm, g)
    process = InteractiveIsing.StatefulAlgorithms.Process(algorithm, process_inputs...; repeat = 1)
    push!(processes(g), process)
    context_update = (; Metropolis_1 = (; T = Float32(3.0)))
    InteractiveIsing.StatefulAlgorithms.context(
        process,
        InteractiveIsing.StatefulAlgorithms.merge_into_subcontexts(InteractiveIsing.StatefulAlgorithms.context(process), context_update),
    )
    Windows._poll!(host)
    @test temperature_panel[:slider].value[] ≈ 3.0f0
    @test temp(g) ≈ 3.0f0

    temp!(g, 4.0f0)
    Windows._poll!(host)
    @test temperature_panel[:slider].value[] ≈ 4.0f0
    @test temp(g) ≈ 4.0f0
    Windows._poll!(host)
    @test temperature_panel[:slider].value[] ≈ 4.0f0
    @test temp(g) ≈ 4.0f0

    temperature_panel[:slider].value[] = 1.5f0
    @test temp(g) ≈ 1.5f0
    @test getproperty(getproperty(StatefulAlgorithms.context(process), :Metropolis_1), :T) ≈ 1.5f0

    close(host)
    @test host.closed
    @test graph_state_obs[] isa graph_state_type
    @test size(graph_state_obs[]) == (0, 0)
end

@testset "SimulationPanel hidden left buttons" begin
    g = IsingGraph(3, 3, Continuous(), StateSet(-1.0f0, 1.0f0); precision = Float32)
    host = Windows.WindowHost(Figure(); screen = nothing, fps = 30, polling_rate = 10)
    handle = Windows.panel!(host, Windows.SimulationPanel(g; hide_left_buttons = true), (1, 1))
    parameter_panel = handle.children[:hamiltonian_parameters]

    @test handle.panel.hide_left_buttons
    @test !parameter_panel.panel.show_buttons
    @test parameter_panel[:buttons_hidden]
    @test isempty(parameter_panel[:selector_buttons])
    @test haskey(parameter_panel, :display_axis)
    @test Windows.hasaxis(parameter_panel)
    @test Windows.tofigure(handle) isa Figure

    state(g[1]) .= 0.75f0
    Windows._tick!(host)
    @test all(parameter_panel[:display_obs][] .== 0.75f0)

    close(host)
    @test host.closed
end

@testset "SimulationPanel interactive variable sliders" begin
    g = IsingGraph(3, 3, Continuous(), StateSet(-1.0f0, 1.0f0); precision = Float32)
    interactivevar!(g, LocalLangevin, :stepsize; value = 0.05f0, range = 0.0:0.01:0.2)

    host = Windows.WindowHost(Figure(); screen = nothing, fps = 30, polling_rate = 10)
    handle = Windows.panel!(host, Windows.SimulationPanel(g), (1, 1))
    @test haskey(handle.children, :interactive_variables)

    interactive_panel = handle.children[:interactive_variables]
    @test length(interactive_panel[:entries]) == 1
    entry = only(interactive_panel[:entries])
    @test entry.slider.value[] ≈ 0.05f0
    @test entry.delta[] ≈ 0.01f0

    proc = createProcess(g, LocalLangevin(); lifetime = 1)
    wait(proc)
    stepsize = getproperty(getproperty(StatefulAlgorithms.getcontext(proc), :LocalLangevin_1), :stepsize)
    @test stepsize[] ≈ 0.05f0

    entry.apply_step!(1)
    @test stepsize[] ≈ 0.06f0

    entry.delta_textbox.displayed_string[] = "0.02"
    entry.apply_step!(1)
    @test entry.delta[] ≈ 0.02f0
    @test stepsize[] ≈ 0.08f0

    entry.apply_step!(-1)
    @test stepsize[] ≈ 0.06f0

    entry.slider.value[] = 0.1f0
    @test stepsize[] ≈ 0.1f0
    @test only(interactivevars(g)).value ≈ 0.1f0

    close(host)
    @test host.closed
end

@testset "SimulationPanel conditional optional panels" begin
    g_langevin = IsingGraph(3, 3, Continuous(), StateSet(-1.0f0, 1.0f0); precision = Float32)
    host_langevin = Windows.WindowHost(Figure(); screen = nothing, fps = 30, polling_rate = 10)
    handle_langevin = Windows.panel!(host_langevin, Windows.SimulationPanel(g_langevin), (1, 1))
    @test !haskey(handle_langevin.children, :kinetic_time)
    @test haskey(handle_langevin.children, :temperature)
    close(host_langevin)
    @test host_langevin.closed

    g_kinetic = IsingGraph(3, 3, Continuous(), StateSet(-1.0f0, 1.0f0); precision = Float32)
    g_kinetic.default_algorithm = KineticMC()
    host_kinetic = Windows.WindowHost(Figure(); screen = nothing, fps = 30, polling_rate = 10)
    handle_kinetic = Windows.panel!(host_kinetic, Windows.SimulationPanel(g_kinetic), (1, 1))
    @test haskey(handle_kinetic.children, :kinetic_time)
    @test haskey(handle_kinetic.children, :temperature)
    close(host_kinetic)
    @test host_kinetic.closed
end

@testset "SimulationPanel layer selector construction" begin
    g = IsingGraph(
        Layer(3, 3, Continuous(), StateSet(-1.0f0, 1.0f0)),
        Layer(3, 3, Continuous(), StateSet(-1.0f0, 1.0f0));
        precision = Float32,
    )
    host = Windows.WindowHost(Figure(); screen = nothing, fps = 30, polling_rate = 10)
    handle = Windows.panel!(host, Windows.SimulationPanel(g), (1, 1))

    @test haskey(handle.children[:status].children, :layer_selector)

    close(host)
    @test host.closed
end

@testset "3D topology coordinates use hexagonal lattice layout" begin
    row_spacing = sqrt(3.0f0) / 2
    square_top = SquareTopology(
        (2, 2, 2);
        lattice_constants = (2.0f0, 3.0f0, 4.0f0),
        periodic = false,
    )
    square_xs, square_ys, square_zs = Windows._coordinates_3d!(nothing, square_top, size(square_top))

    @test square_xs == [1, 2, 1, 2, 1, 2, 1, 2]
    @test square_ys == [1, 1, 2, 2, 1, 1, 2, 2]
    @test square_zs == [1, 1, 1, 1, 2, 2, 2, 2]

    top = sizeto(
        LatticeTopology(
            (0.0f0, row_spacing, 0.0f0),
            (1.0f0, 0.0f0, 0.0f0),
            (0.0f0, 0.0f0, 1.0f0);
            layout = ZigZagRows(),
            periodic = false,
            lattice_type = Hexagonal,
        ),
        (3, 3, 2),
    )

    xs, ys, zs = Windows._coordinates_3d!(nothing, top, size(top))
    linear = LinearIndices(size(top))
    first_row = linear[CartesianIndex(1, 1, 1)]
    staggered_row = linear[CartesianIndex(2, 1, 1)]

    @test xs[first_row] ≈ 1.0f0
    @test ys[first_row] ≈ row_spacing
    @test zs[first_row] ≈ 1.0f0
    @test xs[staggered_row] ≈ 1.5f0
    @test ys[staggered_row] ≈ 2row_spacing
    @test zs[staggered_row] ≈ 1.0f0
end

@testset "3D SimulationPanel figure construction" begin
    wg = @WG window_test_weights NN = 1
    g = IsingGraph(
        4, 4, 3,
        Continuous(),
        wg,
        LatticeConstants(1.0f0, 1.0f0, 1.0f0),
        StateSet(-1.5f0, 1.5f0),
        Ising(c = ConstVal(0.0f0), b = 0) + CoulombHamiltonian(recalc = 1),
        periodic = (:x, :y),
    )
    host = Windows.WindowHost(Figure(); screen = nothing, fps = 30, polling_rate = 10)
    handle = Windows.panel!(host, Windows.SimulationPanel(g), (1, 1))
    coulomb = InteractiveIsing.gethamiltonian(g.hamiltonian, CoulombHamiltonian)
    parameter_panel = handle.children[:hamiltonian_parameters]
    entries = parameter_panel[:entries]
    labels = getfield.(entries, :term_label)
    graph_state_obs = parameter_panel[:display_obs]
    graph_state_type = typeof(graph_state_obs[])

    @test :u in getfield.(entries, :name)
    @test Symbol("ρ") in getfield.(entries, :name)
    @test :localpotential ∉ getfield.(entries, :name)
    @test any(occursin("CoulombHamiltonian", label) for label in labels)
    @test all(!occursin("CoulombInternal", label) for label in labels)
    @test all(!occursin("Parameters", label) for label in labels)
    @test haskey(handle.children, :temperature)
    @test haskey(handle.children[:temperature], :slider)
    @test parameter_panel[:display_is_3d]
    @test length(parameter_panel[:display_obs][]) == nstates(g)
    @test parameter_panel[:display_plot].transform_marker[] == true
    parameter_panel[:display_axis].azimuth[] = 1.1
    parameter_panel[:display_axis].elevation[] = 0.7

    parameter_panel[:selected][] = findfirst(==(:u), getfield.(entries, :name))
    Windows._draw_hamiltonian_entry!(parameter_panel)
    @test parameter_panel[:display_is_3d]
    @test length(parameter_panel[:display_obs][]) == prod(size(coulomb.u))
    @test parameter_panel[:display_plot].transform_marker[] == true
    @test parameter_panel[:display_plot].colorrange[][1] == -parameter_panel[:display_plot].colorrange[][2]
    u_obs = parameter_panel[:display_obs][]
    @test pointer(u_obs) == pointer(coulomb.u)
    Windows._tick!(host)
    @test parameter_panel[:display_obs][] === u_obs
    @test pointer(parameter_panel[:display_obs][]) == pointer(coulomb.u)
    @test parameter_panel[:display_axis].azimuth[] ≈ 1.1
    @test parameter_panel[:display_axis].elevation[] ≈ 0.7

    parameter_panel[:selected][] = findfirst(==(Symbol("ρ")), getfield.(entries, :name))
    Windows._draw_hamiltonian_entry!(parameter_panel)
    @test parameter_panel[:display_is_3d]
    @test length(parameter_panel[:display_obs][]) == prod(size(coulomb.ρ))
    ρ_obs = parameter_panel[:display_obs][]
    ρ_display_obs = parameter_panel[:display_obs]
    ρ_display_type = typeof(ρ_display_obs[])
    @test pointer(ρ_obs) == pointer(coulomb.ρ)
    Windows._tick!(host)
    @test parameter_panel[:display_obs][] === ρ_obs
    @test pointer(parameter_panel[:display_obs][]) == pointer(coulomb.ρ)
    @test parameter_panel[:display_axis].azimuth[] ≈ 1.1
    @test parameter_panel[:display_axis].elevation[] ≈ 0.7

    close(host)
    @test host.closed
    @test graph_state_obs[] isa graph_state_type
    @test length(graph_state_obs[]) == 0
    @test ρ_display_obs[] isa ρ_display_type
    @test length(ρ_display_obs[]) == 0
end
