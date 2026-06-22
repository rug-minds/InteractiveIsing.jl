using InteractiveIsing
using InteractiveIsing.StatefulAlgorithms

defect_hopping_weights(; dr) = dr == 1 ? 1f0 : 0f0

nx, ny, nz = 10, 10, 5
vacancy_linear = zeros(Float32, nx * ny * nz)
vacancy_stiffness = zeros(Float32, nx * ny * nz)
vacancy_field = zeros(Float32, nx * ny * nz)
wg = @WG defect_hopping_weights NN = 1
vacancy_hopping_scale = 180f0
defect_step_interval = 1000

weak_fast_vacancy_hamiltonian =
    ExternalFieldShiftCoupling(0.04f0; hopping_scale = 1f0)

polar_trapping_vacancy_hamiltonian =
    ExternalFieldShiftCoupling(0.08f0; hopping_scale = vacancy_hopping_scale) +
    LocalPotentialShiftCoupling(1, -0.03f0; hopping_scale = vacancy_hopping_scale)

stiffness_trapping_vacancy_hamiltonian =
    LocalPotentialShiftCoupling(2, 0.04f0; hopping_scale = vacancy_hopping_scale)

mixed_charged_vacancy_hamiltonian =
    ExternalFieldShiftCoupling(0.06f0; hopping_scale = vacancy_hopping_scale) +
    LocalPotentialShiftCoupling(2, 0.03f0; hopping_scale = vacancy_hopping_scale)

vacancy_hamiltonian = mixed_charged_vacancy_hamiltonian

g = IsingGraph(
    nx,
    ny,
    nz,
    Continuous(),
    wg,
    LatticeConstants(1f0, 1f0, 1f0),
    StateSet(-1.5f0, 1.5f0),
    Quadratic(c = ConstVal(-1.0f0), localpotential = ConstFill(1f0)) +
        Quartic(c = ConstVal(1.0f0), localpotential = ConstFill(1f0)) +
        PolynomialHamiltonian(1; c = ConstVal(1.0f0), localpotential = vacancy_linear) +
        Quadratic(c = ConstVal(1.0f0), localpotential = vacancy_stiffness) +
        ExtField(b = vacancy_field, c = 1.0f0) +
        Bilinear(),
    periodic = (:x, :y),
    precision = Float32,
    initial_state = 0f0,
)

spin_field = reshape(state(g), nx, ny, nz)
for I in CartesianIndices(spin_field)
    x, y, z = Tuple(I)
    spin_field[I] =
        0.15f0 +
        0.65f0 * Float32(x - 1) / Float32(nx - 1) +
        0.25f0 * sinpi(2f0 * Float32(y - 1) / Float32(ny)) +
        0.04f0 * (Float32(z) - (Float32(nz) + 1f0) / 2f0)^2
end
temp!(g, 0.06f0)

defects = DefectHopping(
    g;
    defects = [
        CartesianIndex(3, 3, 2),
        CartesianIndex(7, 7, 4),
        CartesianIndex(5, 4, 3),
    ],
    hamiltonian = vacancy_hamiltonian,
)

langevin_algorithm = LocalLangevin(stepsize = 0.03f0, adjusted = false)
defect_algorithm = Metropolis()

algorithm = @CompositeAlgorithm begin
    @alias langevin = langevin_algorithm
    @alias defect_metro = defect_algorithm
    @replace langevin.T => defect_metro.T

    langevin()
    @every defect_step_interval defect_metro()
end

graph_host = interface(g; title = "10x10x5 spins with hopping defects")
defect_host = interface(defects; title = "Live defect positions")
process = createProcessManual(
    g,
    algorithm,
    StatefulAlgorithms.Init(:langevin; model = g),
    StatefulAlgorithms.Init(:defect_metro; model = defects),
    StatefulAlgorithms.Interactive(:langevin, :T),
)
