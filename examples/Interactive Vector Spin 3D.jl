using InteractiveIsing
using InteractiveIsing.Processes

function vector_spin_3d_weights(; dr::R) where {R}
    return dr == 1 ? 1f0 : 0f0
end

const VECTOR_SPIN_3D_WG = @WG vector_spin_3d_weights NN = 1

"""
    interactive_vector_spin_3d_example()

Open a three-dimensional bounded vector-spin simulation with 3D vector arrows.
"""
function interactive_vector_spin_3d_example()
    g = VectorSpinGraph(
        48,
        48,
        10,
        Continuous(),
        VECTOR_SPIN_3D_WG,
        LatticeConstants(1f0, 1f0, 1f0),
        StateSet(-1.0f0, 1.0f0),
        VectorSpin() + VectorMagnitudePenalty(c = 10f0, target = 0.65f0),
        VectorSpinLocalProposer(0.08f0);
        dimension = 3,
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
        value = 0.01f0,
        range = 0.0:0.001:0.08,
        label = "stepsize",
    )

    createProcess(g, dynamics = LocalLangevin(stepsize = 0.01f0))
    host = interface(g; framerate = 30, polling_rate = 10, title = "Interactive Vector Spin 3D")
    return host, g
end

interactive_vector_spin_3d_example()
