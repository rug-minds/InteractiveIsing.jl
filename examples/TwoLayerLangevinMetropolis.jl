using InteractiveIsing
using InteractiveIsing.StatefulAlgorithms

# Run with:
#     julia --project examples/TwoLayerLangevinMetropolis.jl
#
# This demonstrates `LayerwiseMC`: a real composite MCAlgorithm that delegates
# each layer to any existing MCAlgorithm through normal StatefulAlgorithms.init/step!.

function make_two_layer_graph()
    continuous = Layer(16, Continuous(), StateSet(-1f0, 1f0); periodic = false)
    binary = Layer(16, Discrete(), StateSet(0f0, 1f0); periodic = false)
    hamiltonian = Ising(c = ConstVal(1f0), localpotential = ConstFill(0.25f0), b = ConstFill(0f0))

    g = IsingGraph(continuous, binary, hamiltonian; precision = Float32)
    temp!(g, 1f0)
    return g
end

function make_layerwise_algorithm()
    return LayerwiseMC(
        1 => LocalLangevin(stepsize = 0.01f0, adjusted = true),
        2 => Metropolis();
        scheduler = SequentialLayerScheduler((2, 4)),
    )
end

function run_example(; nsteps = 500)
    g = make_two_layer_graph()
    algorithm = make_layerwise_algorithm()
    context = StatefulAlgorithms.init(algorithm, (; model = g))

    accepted = 0
    attempted = 0
    for _ in 1:nsteps
        out = StatefulAlgorithms.step!(algorithm, context)
        accepted += out.accepted
        attempted += out.attempted
    end

    return (;
        algorithm = "LayerwiseMC",
        layer1_algorithm = "LocalLangevin",
        layer2_algorithm = "Metropolis",
        attempted,
        acceptance_rate = attempted == 0 ? NaN : accepted / attempted,
        layer1_extrema = extrema(state(g[1])),
        layer2_values = sort(unique(vec(state(g[2])))),
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    display(run_example())
end
