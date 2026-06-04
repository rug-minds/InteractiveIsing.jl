using Test
using InteractiveIsing
using InteractiveIsing.StatefulAlgorithms
using Random

@testset "MC Process Inputs" begin
    g = IsingGraph(2, 2, Continuous(); precision = Float32)

    @test LocalLangevin() isa LocalLangevin{:random}
    @test LocalLangevin() isa LocalLangevin{:random,true}
    @test LocalLangevin(adjusted = false) isa LocalLangevin{:random,false}
    @test LocalLangevin(order = :deterministic) isa LocalLangevin{:deterministic}
    @test LocalLangevin{:random}(adjusted = false) isa LocalLangevin{:random,false}
    @test LocalLangevin{:random,false}() isa LocalLangevin{:random,false}
    @test LocalLangevin(order = :cyclic) isa LocalLangevin{:cyclic}
    @test_throws ArgumentError LocalLangevin(order = :unknown)
    @test_throws ArgumentError LocalLangevin{:random,:unknown}()
    @test BlockLangevin() isa BlockLangevin{false}
    @test BlockLangevin(adjusted = true) isa BlockLangevin{true}
    @test BlockLangevin{true}() isa BlockLangevin{true}
    @test_throws ArgumentError BlockLangevin{:unknown}()
    @test DynamicBlockLangevin() isa DynamicBlockLangevin{256,false}
    @test DynamicBlockLangevin(max_blocksize = 17, adjusted = true) isa DynamicBlockLangevin{17,true}
    @test DynamicBlockLangevin{13}(adjusted = true) isa DynamicBlockLangevin{13,true}
    @test_throws ArgumentError DynamicBlockLangevin{0}()
    @test_throws ArgumentError DynamicBlockLangevin{8,:unknown}()

    step_ctx = StatefulAlgorithms.init(LocalLangevin(adjusted = false, order = :deterministic), (;model = g))
    step_result = StatefulAlgorithms.step!(LocalLangevin(adjusted = false, order = :deterministic), step_ctx)
    @test step_result.attempted == 1

    function single_spin_langevin_change_count(algorithm)
        graph = IsingGraph(
            4,
            Continuous(),
            StateSet(-10f0, 10f0),
            Clamping(1f0, fill(0f0, 4));
            precision = Float32,
            initial_state = 1f0,
        )
        temp!(graph, 0f0)
        context = StatefulAlgorithms.init(algorithm, (;model = graph))
        before = copy(state(graph))
        out = StatefulAlgorithms.step!(algorithm, context)
        after = state(graph)
        return count(i -> before[i] != after[i], eachindex(before)), out
    end

    for algorithm in (
        GlobalLangevin(stepsize = 0.1f0, adjusted = false, group_steps = 8),
        GlobalLangevin(stepsize = 0.1f0, adjusted = true, group_steps = 8),
        BlockLangevin(stepsize = 0.1f0, adjusted = false, block_size = 3, group_steps = 8),
        BlockLangevin(stepsize = 0.1f0, adjusted = true, block_size = 3, group_steps = 8),
        DynamicBlockLangevin(stepsize = 0.1f0, adjusted = false, max_blocksize = 3, group_steps = 8),
        DynamicBlockLangevin(stepsize = 0.1f0, adjusted = true, max_blocksize = 3, group_steps = 8),
    )
        changed, out = single_spin_langevin_change_count(algorithm)
        @test changed == 1
        @test out.proposal isa FlipProposal
        @test out.attempted == 1
        @test out.accepted == 1
    end

    function langevin_refresh_pattern(algorithm, n)
        graph = IsingGraph(
            n,
            Continuous(),
            StateSet(-10f0, 10f0),
            Clamping(1f0, fill(0f0, n));
            precision = Float32,
            initial_state = 1f0,
        )
        temp!(graph, 0f0)
        context = StatefulAlgorithms.init(algorithm, (;model = graph))
        refreshed = Bool[]
        for _ in 1:(n + 1)
            out = StatefulAlgorithms.step!(algorithm, context)
            context = merge(context, out)
            push!(refreshed, out.refreshed_gradient)
        end
        return refreshed
    end

    @test langevin_refresh_pattern(GlobalLangevin(stepsize = 0.1f0, adjusted = false), 4) ==
        [true, false, false, false, true]
    @test langevin_refresh_pattern(BlockLangevin(stepsize = 0.1f0, adjusted = false, block_size = 3), 3) ==
        [true, false, false, true]
    @test langevin_refresh_pattern(GlobalLangevin(stepsize = 0.1f0, adjusted = true), 4) ==
        [true, false, false, false, true]
    @test langevin_refresh_pattern(BlockLangevin(stepsize = 0.1f0, adjusted = true, block_size = 3), 3) ==
        [true, false, false, true]

    @test InteractiveIsing._langevin_boundary_drift_step(1f0, -1f0, -1f0, 1f0) == 0f0
    @test InteractiveIsing._langevin_boundary_drift_step(-1f0, 1f0, -1f0, 1f0) == 0f0
    @test InteractiveIsing._langevin_boundary_drift_step(-1f0, -1f0, -1f0, 1f0) == -1f0
    @test InteractiveIsing._langevin_boundary_drift_step(1f0, 1f0, -1f0, 1f0) == 1f0

    function unadjusted_langevin_boundary_state(algorithm)
        graph = IsingGraph(
            1,
            Continuous(),
            StateSet(-1f0, 1f0),
            Clamping(1f0, [10f0], [1f0]);
            precision = Float32,
            initial_state = 0.9f0,
        )
        temp!(graph, 0f0)
        context = StatefulAlgorithms.init(algorithm, (;model = graph))
        out = StatefulAlgorithms.step!(algorithm, context)
        return state(graph)[1], out
    end

    for algorithm in (
        LocalLangevin(stepsize = 0.1f0, adjusted = false, order = :deterministic),
        GlobalLangevin(stepsize = 0.1f0, adjusted = false),
        BlockLangevin(stepsize = 0.1f0, adjusted = false, block_size = 1),
        DynamicBlockLangevin(stepsize = 0.1f0, adjusted = false, max_blocksize = 1),
    )
        boundary_state, out = unadjusted_langevin_boundary_state(algorithm)
        @test boundary_state == 1f0
        @test out.accepted == 1
        @test out.reflected_fraction == 0f0
    end

    function ordered_block_shuffle_groups()
        graph = IsingGraph(
            8,
            Continuous(),
            StateSet(-10f0, 10f0),
            Clamping(1f0, fill(0f0, 8));
            precision = Float32,
            initial_state = 1f0,
        )
        temp!(graph, 0f0)
        algorithm = BlockLangevin(stepsize = 0.1f0, adjusted = false, block_size = 3)
        context = StatefulAlgorithms.init(algorithm, (;model = graph))
        Random.seed!(context.rng, 1)
        out = StatefulAlgorithms.step!(algorithm, context)
        first = copy(@view context.block_idxs[1:out.block_size])
        for _ in 2:out.block_size
            out = StatefulAlgorithms.step!(algorithm, context)
            context = merge(context, out)
        end
        out = StatefulAlgorithms.step!(algorithm, context)
        second = copy(@view context.block_idxs[1:out.block_size])
        return first, second, out
    end

    first_block, second_block, second_out = ordered_block_shuffle_groups()
    @test second_out.refreshed_gradient
    @test length(unique(first_block)) == length(first_block)
    @test length(unique(second_block)) == length(second_block)
    @test isempty(intersect(first_block, second_block))

    function adjusted_langevin_acceptance(algorithm)
        graph = IsingGraph(
            8,
            Continuous(),
            StateSet(-10f0, 10f0),
            Clamping(1f0, fill(0f0, 8));
            precision = Float32,
            initial_state = 1f0,
        )
        temp!(graph, 1f0)
        context = StatefulAlgorithms.init(algorithm, (;model = graph))
        Random.seed!(context.rng, 1)
        before = copy(state(graph))
        accepted = 0
        for _ in 1:32
            out = StatefulAlgorithms.step!(algorithm, context)
            context = merge(context, out)
            accepted += out.accepted
        end
        changed = count(i -> before[i] != state(graph)[i], eachindex(before))
        return accepted, changed
    end

    for algorithm in (
        GlobalLangevin(stepsize = 0.01f0, max_drift_fraction = 0.15f0, adjusted = true),
        BlockLangevin(stepsize = 0.01f0, max_drift_fraction = 0.15f0, adjusted = true, block_size = 4),
    )
        accepted, changed = adjusted_langevin_acceptance(algorithm)
        @test accepted > 0
        @test changed > 0
    end

    zero_temp_graph = IsingGraph(
        1,
        1,
        Continuous(),
        StateSet(-10f0, 10f0),
        Clamping(1f0, [0f0], [1f0]);
        precision = Float32,
        initial_state = 1f0,
    )
    temp!(zero_temp_graph, 0f0)
    zero_temp_alg = LocalLangevin(stepsize = 2.5f0, adjusted = false, order = :deterministic)
    zero_temp_ctx = StatefulAlgorithms.init(zero_temp_alg, (;model = zero_temp_graph))
    zero_temp_vals = Float32[]
    for _ in 1:8
        zero_temp_out = StatefulAlgorithms.step!(zero_temp_alg, zero_temp_ctx)
        zero_temp_ctx = merge(zero_temp_ctx, zero_temp_out)
        push!(zero_temp_vals, state(zero_temp_graph)[1])
    end
    @test all(>=(0f0), zero_temp_vals)
    @test issorted(abs.(zero_temp_vals); rev = true)

    loop = deepcopy(CompositeAlgorithm(Unique(Metropolis()), Unique(LocalLangevin())))
    loop_inputs = InteractiveIsing._mc_model_inits(loop, g)
    loop_process = Process(loop, loop_inputs...; repeats = 1)
    loop_context = getcontext(loop_process)

    @test loop_context[loop[1]].model === g
    @test loop_context[loop[2]].model === g
    loop_algorithm = StatefulAlgorithms.getalgo(loop_process)
    @test StatefulAlgorithms.getoptions(loop_algorithm) == StatefulAlgorithms.getoptions(StatefulAlgorithms.getplan(loop_algorithm))
    @test StatefulAlgorithms.get_shares(loop_algorithm) == StatefulAlgorithms.get_shares(StatefulAlgorithms.getplan(loop_algorithm))
    @test StatefulAlgorithms.get_routes(loop_algorithm) == StatefulAlgorithms.get_routes(StatefulAlgorithms.getplan(loop_algorithm))
    @test StatefulAlgorithms.getstates(loop_algorithm) == ()
    @test StatefulAlgorithms.getalgos(loop_algorithm) == StatefulAlgorithms.getalgos(StatefulAlgorithms.getplan(loop_algorithm))
    @test keys(loop_algorithm) == keys(StatefulAlgorithms.getplan(loop_algorithm))

    kinetic = deepcopy(KineticMC())
    kinetic_inputs = InteractiveIsing._mc_model_inits(kinetic, g)
    kinetic_process = Process(kinetic, kinetic_inputs...; repeats = 1)
    kinetic_context = getcontext(kinetic_process)

    @test kinetic_context[kinetic].model === g

    runtime_loop = CompositeAlgorithm(Unique(Metropolis()), Unique(LocalLangevin()))
    runtime_loop_process = createProcess(g, runtime_loop; lifetime = 1)
    wait(runtime_loop_process)
    @test istaskdone(runtime_loop_process.task)
    @test length(processes(g)) == 1

    runtime_kinetic_process = createProcess(g, KineticMC(); lifetime = 1)
    wait(runtime_kinetic_process)
    @test istaskdone(runtime_kinetic_process.task)
    @test length(processes(g)) == 1

    extra_process = createProcess(g, Metropolis(); lifetime = 1, allow_multiple = true)
    wait(extra_process)
    @test istaskdone(extra_process.task)
    @test length(processes(g)) == 2

    default_lifetime_process = createProcess(g, Metropolis())
    try
        @test StatefulAlgorithms.lifetime(default_lifetime_process) isa StatefulAlgorithms.Indefinite
        @test length(processes(g)) == 1
    finally
        StatefulAlgorithms.close(g)
    end
end

@testset "Interactive temperature vars" begin
    g = IsingGraph(4, 4, Continuous(); precision = Float32)
    g.addons[:interactive] = true

    metro_proc = createProcess(g, Metropolis(); lifetime = 1)
    wait(metro_proc)
    metro_subcontexts = StatefulAlgorithms.get_subcontexts(getcontext(metro_proc))
    metro_key = only(filter(name -> name !== :globals, propertynames(metro_subcontexts)))
    metro_T = getproperty(StatefulAlgorithms.getdata(getproperty(metro_subcontexts, metro_key)), :T)
    @test metro_T isa StatefulAlgorithms.InteractiveVar{Float32}

    local_proc = createProcess(g, LocalLangevin(); lifetime = 1)
    wait(local_proc)
    local_subcontexts = StatefulAlgorithms.get_subcontexts(getcontext(local_proc))
    local_key = only(filter(name -> name !== :globals, propertynames(local_subcontexts)))
    local_T = getproperty(StatefulAlgorithms.getdata(getproperty(local_subcontexts, local_key)), :T)
    @test local_T isa StatefulAlgorithms.InteractiveVar{Float32}

    global_proc = createProcess(g, GlobalLangevin(); lifetime = 1)
    wait(global_proc)
    global_subcontexts = StatefulAlgorithms.get_subcontexts(getcontext(global_proc))
    global_key = only(filter(name -> name !== :globals, propertynames(global_subcontexts)))
    global_T = getproperty(StatefulAlgorithms.getdata(getproperty(global_subcontexts, global_key)), :T)
    @test global_T isa StatefulAlgorithms.InteractiveVar{Float32}

    block_proc = createProcess(g, BlockLangevin(); lifetime = 1)
    wait(block_proc)
    block_subcontexts = StatefulAlgorithms.get_subcontexts(getcontext(block_proc))
    block_key = only(filter(name -> name !== :globals, propertynames(block_subcontexts)))
    block_T = getproperty(StatefulAlgorithms.getdata(getproperty(block_subcontexts, block_key)), :T)
    @test block_T isa StatefulAlgorithms.InteractiveVar{Float32}
end

@testset "Interactive Langevin parameter vars" begin
    g = IsingGraph(4, 4, Continuous(); precision = Float32)
    interactivevar!(g, LocalLangevin, :stepsize; value = 0.025f0, range = 0.0:0.005:0.2)

    proc = createProcess(g, LocalLangevin(); lifetime = 1)
    wait(proc)
    subcontexts = StatefulAlgorithms.get_subcontexts(getcontext(proc))
    key = only(filter(name -> name !== :globals, propertynames(subcontexts)))
    stepsize = getproperty(StatefulAlgorithms.getdata(getproperty(subcontexts, key)), :stepsize)
    @test stepsize isa StatefulAlgorithms.InteractiveVar{Float32}
    @test stepsize[] ≈ 0.025f0
end

@testset "Index Set API" begin
    toggled = ToggledIndexSet(1:2, 3:5)
    @test collect(InteractiveIsing.sampling_indices(toggled)) == [1, 2, 3, 4, 5]
    @test !InteractiveIsing.consume_changed!(toggled)

    InteractiveIsing.off!(toggled, 2)
    @test collect(InteractiveIsing.sampling_indices(toggled)) == [1, 2]
    @test InteractiveIsing.consume_changed!(toggled)
    @test !InteractiveIsing.consume_changed!(toggled)

    layered = ToggledLayerIndexSet(2, 1:2, 3:4, 5:6)
    @test collect(InteractiveIsing.sampling_indices(layered)) == [3, 4]
    @test !InteractiveIsing.consume_changed!(layered)

    InteractiveIsing.off!(layered)
    @test collect(InteractiveIsing.sampling_indices(layered)) == [1, 2, 5, 6]
    @test InteractiveIsing.consume_changed!(layered)
    @test !InteractiveIsing.consume_changed!(layered)

    g = IsingGraph(
        Layer(2, Continuous(), Coords(0, 1, 0)),
        Layer(2, Continuous(), Coords(0, 2, 0));
        precision = Float32,
        index_set = graph -> ToggledIndexSet(graph),
    )
    @test index_set(g) isa ToggledIndexSet
    @test collect(sampling_indices(g)) == [1, 2, 3, 4]

    InteractiveIsing.off!(index_set(g), 2)
    @test collect(sampling_indices(g)) == [1, 2]
    @test InteractiveIsing.consume_changed!(g)
    @test !InteractiveIsing.consume_changed!(g)

    proc = createProcess(g, LocalLangevin(adjusted = false); lifetime = 1)
    wait(proc)
    @test istaskdone(proc.task)

    deterministic_proc = createProcess(g, LocalLangevin(order = :deterministic, adjusted = false); lifetime = 1)
    wait(deterministic_proc)
    @test istaskdone(deterministic_proc.task)

    global_proc = createProcess(g, GlobalLangevin(adjusted = false); lifetime = 1)
    wait(global_proc)
    @test istaskdone(global_proc.task)
end
