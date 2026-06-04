using Test
using InteractiveIsing
using InteractiveIsing.StatefulAlgorithms

function layerwise_test_graph()
    continuous = Layer(8, Continuous(), StateSet(-1f0, 1f0); periodic = false)
    binary = Layer(8, Discrete(), StateSet(0f0, 1f0); periodic = false)
    hamiltonian = Ising(c = ConstVal(1f0), localpotential = ConstFill(0.25f0), b = ConstFill(0f0))
    g = IsingGraph(continuous, binary, hamiltonian; precision = Float32)
    temp!(g, 1f0)
    return g
end

@testset "LayerwiseMC" begin
    g = layerwise_test_graph()
    original_index_set = getfield(g, :index_set)
    algorithm = LayerwiseMC(
        1 => LocalLangevin(stepsize = 0.01f0, adjusted = true),
        2 => Metropolis();
        scheduler = SequentialLayerScheduler((2, 3)),
    )

    context = StatefulAlgorithms.init(algorithm, (; model = g))

    @test getfield(g, :index_set) == original_index_set
    @test context.layer_index_sets isa Tuple
    @test context.subcontexts isa Tuple
    @test collect(context.subcontexts[1][].active_spins) == collect(InteractiveIsing.graphidxs(g[1]))
    @test collect(context.subcontexts[2][].proposer.index_set) == collect(InteractiveIsing.graphidxs(g[2]))

    out = StatefulAlgorithms.step!(algorithm, context)

    @test getfield(g, :index_set) == original_index_set
    @test out.attempted == 5
    @test 0 <= out.accepted <= out.attempted
    @test all(isfinite, vec(state(g[1])))
    @test all(x -> x == 0f0 || x == 1f0, vec(state(g[2])))
end
