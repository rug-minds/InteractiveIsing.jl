using Test
using InteractiveIsing
using InteractiveIsing.Processes

@testset "MC Process Inputs" begin
    g = IsingGraph(2, 2, Continuous(); precision = Float32)

    loop = deepcopy(SimpleAlgo(Unique(Metropolis()), Unique(LangevinDynamics())))
    loop_inputs = InteractiveIsing._merge_graph_inputs(loop, g)
    loop_process = Process(loop, loop_inputs...; lifetime = 1)
    loop_context = getcontext(loop_process)

    @test loop_context[loop[1]].state === g
    @test loop_context[loop[2]].state === g

    kinetic = deepcopy(KineticMC())
    kinetic_inputs = InteractiveIsing._merge_graph_inputs(kinetic, g)
    kinetic_process = Process(kinetic, kinetic_inputs...; lifetime = 1)
    kinetic_context = getcontext(kinetic_process)

    @test kinetic_context[kinetic].state === g
    @test kinetic_context[kinetic].structure === g

    runtime_loop = SimpleAlgo(Unique(Metropolis()), Unique(LangevinDynamics()))
    runtime_loop_process = createProcess(g, runtime_loop; lifetime = 1)
    wait(runtime_loop_process)
    @test istaskdone(runtime_loop_process.task)

    runtime_kinetic_process = createProcess(g, KineticMC(); lifetime = 1)
    wait(runtime_kinetic_process)
    @test istaskdone(runtime_kinetic_process.task)
end
