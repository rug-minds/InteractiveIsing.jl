using InteractiveIsing
using InteractiveIsing.StatefulAlgorithms

"""
    make_defect_hopping_graph()

Build a continuous graph with mutable local-potential storage and a defect
hopping proposer for Metropolis.
"""
function make_defect_hopping_graph()
    layer = Layer(16, 16, Continuous(), StateSet(-1.5f0, 1.5f0); periodic = (:x, :y))
    localpotential = zeros(Float32, 16 * 16)
    hamiltonian = Quadratic(c = ConstVal(0.25f0), localpotential = localpotential) +
        Quartic(c = ConstVal(1.0f0), localpotential = copy(localpotential))
    proposer = DefectHopping(
        defects = [CartesianIndex(4, 4), CartesianIndex(12, 12)],
        charge = 0.5f0,
    )

    g = IsingGraph(layer, hamiltonian, proposer; precision = Float32)
    temp!(g, 1.0f0)
    return g
end

"""
    make_defect_hopping_algorithm()

Schedule Langevin spin updates together with slower Metropolis defect hops.
"""
function make_defect_hopping_algorithm()
    return CompositeAlgorithm(
        LocalLangevin(stepsize = 0.02f0, adjusted = false),
        Metropolis(),
        (1, 16),
    )
end

"""
    run_defect_hopping_example(; nsteps=200)

Run a small synchronous defect-hopping simulation and return basic diagnostics.
"""
function run_defect_hopping_example(; nsteps = 200)
    g = make_defect_hopping_graph()
    algorithm = make_defect_hopping_algorithm()
    process = InlineProcess(algorithm, InteractiveIsing._mc_model_inits(algorithm, g)...; repeats = nsteps)
    run(process)

    quadratic = InteractiveIsing.gethamiltonian(g.hamiltonian, InteractiveIsing.PolynomialHamiltonian{2})
    return (;
        state_extrema = extrema(state(g)),
        total_defect_charge = sum(quadratic.lp),
        active_indices_unchanged = collect(sampling_indices(g)) == collect(1:nstates(g)),
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    display(run_defect_hopping_example())
end
