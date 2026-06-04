using InteractiveIsing
using InteractiveIsing.StatefulAlgorithms

function isingweights(; dr::R) where {R}
    return dr == 1 ? 1f0 : 0f0
end

const LANGEVIN_WG = @WG isingweights NN = 1

"""
    interactive_langevin_example()

Open a small continuous-spin Langevin simulation with interactive controls for
temperature, Langevin step size, and drift cap.
"""
function interactive_langevin_example()
    g = IsingGraph(
        48,
        48,
        Continuous(),
        LANGEVIN_WG,
        StateSet(-1.0f0, 1.0f0),
        Ising(c = ConstVal(0f0), b = 0f0, localpotential = 0f0);
        precision = Float32,
        periodic = (:x, :y),
    )

    g.default_algorithm = LocalLangevin(
        stepsize = 0.05f0,
        max_drift_fraction = 0.2f0,
        adjusted = true,
    )
    g.addons[:interactive] = true
    temp!(g, 0.15f0)

    # The SimulationPanel will automatically add sliders plus +/- delta controls
    # for these graph-level interactive variable specs.
    interactivevar!(
        g,
        LocalLangevin,
        :stepsize;
        value = 0.05f0,
        range = 0.0:0.0025:0.25,
        label = "stepsize",
    )
    interactivevar!(
        g,
        LocalLangevin,
        :max_drift_fraction;
        value = 0.2f0,
        range = 0.0:0.01:1.0,
        label = "drift cap",
    )

    interface(g; framerate = 30, polling_rate = 10)
    createProcess(g)
    return nothing
end

interactive_langevin_example()
