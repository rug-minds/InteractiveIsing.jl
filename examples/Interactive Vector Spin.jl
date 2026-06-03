using InteractiveIsing
using InteractiveIsing.Processes
using StaticArrays

function vector_spin_weights(; dr::R) where {R}
    return dr == 1 ? 1f0 : 0f0
end

const VECTOR_SPIN_WG = @WG vector_spin_weights NN = 1

"""
    interactive_vector_spin_example()

Open a two-dimensional bounded vector-spin simulation.
"""
function interactive_vector_spin_example()
    g = VectorSpinGraph(
        48,
        48,
        Continuous(),
        VECTOR_SPIN_WG,
        StateSet(-1.0f0, 1.0f0),
        VectorSpin() + VectorMagnitudePenalty(c = 10f0, target = 0.65f0),
        VectorSpinLocalProposer(0.12f0);
        dimension = 2,
        precision = Float32,
        periodic = (:x, :y),
        unit_norm = false,
    )

    temp!(g, 0.55f0)
    g.addons[:interactive] = true
    interactivevar!(
        g,
        LocalLangevin,
        :stepsize;
        value = 0.0f0,
        range = 0.0:0.001:0.1,
        label = "stepsize",
    )

    createProcess(g, dynamics = LocalLangevin(stepsize = 0.0f0))
    host = interface(g; framerate = 30, polling_rate = 10, title = "Interactive Vector Spin")
    return host, g
end

interactive_vector_spin_example()
