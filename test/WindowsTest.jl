using Test
using GLMakie
using InteractiveIsing
using InteractiveIsing.Windows

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
    @test !Processes.ispaused(host.frame_timer)
    @test !Processes.ispaused(host.poll_timer)
    resume!(host)
    @test !host.paused[]
    close(host)
    @test resource.closed[]
    @test host.closed

    close(host)
    @test resource.closed[]

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
    @test take!(panel_done) == :panel
    @test panel_owner[] === callback_handle

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

@testset "InteractiveLinesPanel and ContextLinesPanel figure construction" begin
    ctx = InteractiveIsing.Processes.ProcessContext(
        (;
            demo = InteractiveIsing.Processes.SubContext(:demo, (; x = collect(1:5), y = collect(2:2:6)), (), ()),
            globals = (;),
        ),
        InteractiveIsing.Processes.NameSpaceRegistry(),
    )
    host = Windows.WindowHost(Figure(); screen = nothing, fps = 30, polling_rate = 10)
    handle = Windows.panel!(
        host,
        Windows.ContextLinesPanel(
            ctx,
            :demo => :x,
            InteractiveIsing.Processes.Var(:demo, :y);
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
    process_inputs = InteractiveIsing._merge_graph_inputs(algorithm, g)
    process = InteractiveIsing.Processes.Process(algorithm, process_inputs...; repeat = 1)
    push!(processes(g), process)
    context_update = (; Metropolis_1 = (; T = Float32(3.0)))
    InteractiveIsing.Processes.context(
        process,
        InteractiveIsing.Processes.merge_into_subcontexts(getfield(process, :context), context_update),
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
    @test getproperty(getproperty(process.context, :Metropolis_1), :T) ≈ 1.5f0

    close(host)
    @test host.closed
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
    parameter_panel[:display_axis].azimuth[] = 1.1
    parameter_panel[:display_axis].elevation[] = 0.7

    parameter_panel[:selected][] = findfirst(==(:u), getfield.(entries, :name))
    Windows._draw_hamiltonian_entry!(parameter_panel)
    @test parameter_panel[:display_is_3d]
    @test length(parameter_panel[:display_obs][]) == prod(size(coulomb.u))
    @test parameter_panel[:display_axis].azimuth[] ≈ 1.1
    @test parameter_panel[:display_axis].elevation[] ≈ 0.7

    parameter_panel[:selected][] = findfirst(==(Symbol("ρ")), getfield.(entries, :name))
    Windows._draw_hamiltonian_entry!(parameter_panel)
    @test parameter_panel[:display_is_3d]
    @test length(parameter_panel[:display_obs][]) == prod(size(coulomb.ρ))
    @test parameter_panel[:display_axis].azimuth[] ≈ 1.1
    @test parameter_panel[:display_axis].elevation[] ≈ 0.7

    close(host)
    @test host.closed
end
