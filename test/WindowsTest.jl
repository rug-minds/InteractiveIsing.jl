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

@testset "ContextLinesPanel figure construction" begin
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
        ),
        (1, 1),
    )

    @test String(handle[:axis].xlabel[]) == "x"
    @test String(handle[:axis].ylabel[]) == "y"
    @test String(handle[:axis].title[]) == "tracked"
    @test length(handle[:x_obs][]) == 3
    @test length(handle[:y_obs][]) == 3
    @test collect(handle[:x_obs][]) == [1, 2, 3]
    @test collect(handle[:y_obs][]) == [2, 4, 6]

    ctx.demo.x[1] = 10
    push!(ctx.demo.y, 8, 10, 12)
    Windows._tick!(host)
    @test length(handle[:x_obs][]) == 5
    @test length(handle[:y_obs][]) == 5
    @test collect(handle[:x_obs][]) == [10, 2, 3, 4, 5]
    @test collect(handle[:y_obs][]) == [2, 4, 6, 8, 10]

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
    @test !isempty(entries)
    @test :state in getfield.(entries, :name)
    @test :lp in getfield.(entries, :name)
    @test :b in getfield.(entries, :name)
    @test :c ∉ getfield.(entries, :name)
    @test :J ∉ getfield.(entries, :name)
    @test any(occursin("Quadratic{", label) for label in labels)
    @test any(occursin("MagField{", label) for label in labels)
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

    temperature_panel[:slider].value[] = 1.5f0
    @test temp(g) ≈ 1.5f0
    @test getproperty(getproperty(process.context, :Metropolis_1), :T) ≈ 1.5f0

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
    @test any(occursin("CoulombHamiltonian{", label) for label in labels)
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
