using InteractiveIsing
using InteractiveIsing.StatefulAlgorithms

"""
    coulomb_defect_weights(; dr)

Nearest-neighbor coupling used by the bilinear part of the 3D graph.
"""
coulomb_defect_weights(; dr) = dr == 1 ? 1f0 : 0f0

nx, ny, nz = 40, 40, 10
wg = @WG coulomb_defect_weights NN = 1

vacancy_stiffness = zeros(Float32, nx * ny * nz)
vacancy_quartic = zeros(Float32, nx * ny * nz)

stepsize = 0.025f0
temperature = 0.04f0
defect_step_interval = 1000

coulomb_scaling = 0.25f0
coulomb_screening = 8f0
coulomb_recalc_interval = 500
vacancy_charge = 0.08f0
electron_charge = 0.04f0
external_field_z = 0.7f0

charged_oxygen_vacancy = (
    CoulombChargeShift(vacancy_charge; split = 0.5f0),
    ExternalFieldBias((0f0, 0f0, external_field_z)),
    LocalPotentialShift(2, 0.025f0),
    LocalPotentialShift(4, 0.008f0),
)

mobile_electron = (
    CoulombChargeShift(-electron_charge; split = 0.5f0),
    ExternalFieldBias((0f0, 0f0, external_field_z)),
)

vacancy_effects = charged_oxygen_vacancy
electron_effects = mobile_electron

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
        Quadratic(c = ConstVal(1.0f0), localpotential = vacancy_stiffness) +
        Quartic(c = ConstVal(1.0f0), localpotential = vacancy_quartic) +
        Bilinear() +
        CoulombHamiltonian(
            scaling = coulomb_scaling,
            screening = coulomb_screening,
            recalc = coulomb_recalc_interval,
            q_positive = vacancy_charge,
            q_negative = electron_charge,
            free_charge_split = 0.5f0,
        ) + MagField(),
    periodic = (:x, :y),
    precision = Float32,
    initial_state = 0f0,
)

spin_field = reshape(state(g), nx, ny, nz)
for I in CartesianIndices(spin_field)
    x, y, z = Tuple(I)
    spin_field[I] =
        0.2f0 * sinpi(2f0 * Float32(x - 1) / Float32(nx)) +
        0.2f0 * cospi(2f0 * Float32(y - 1) / Float32(ny)) +
        0.15f0 * (Float32(z) - (Float32(nz) + 1f0) / 2f0)
end
temp!(g, temperature)

vacancy_positions = [
    CartesianIndex(6, 6, 2),
    CartesianIndex(14, 10, 4),
    CartesianIndex(22, 8, 6),
    CartesianIndex(32, 12, 8),
    CartesianIndex(9, 24, 3),
    CartesianIndex(18, 28, 5),
    CartesianIndex(27, 30, 7),
    CartesianIndex(35, 25, 9),
    CartesianIndex(11, 36, 2),
    CartesianIndex(24, 35, 6),
]

electron_positions = [
    CartesianIndex(4, 5, 1),
    CartesianIndex(8, 7, 3),
    CartesianIndex(12, 9, 4),
    CartesianIndex(16, 11, 5),
    CartesianIndex(20, 7, 6),
    CartesianIndex(24, 9, 7),
    CartesianIndex(28, 11, 8),
    CartesianIndex(34, 13, 9),
    CartesianIndex(7, 22, 2),
    CartesianIndex(11, 26, 4),
    CartesianIndex(15, 30, 5),
    CartesianIndex(19, 34, 6),
    CartesianIndex(23, 28, 7),
    CartesianIndex(29, 32, 8),
    CartesianIndex(33, 26, 9),
    CartesianIndex(37, 24, 10),
    CartesianIndex(9, 38, 1),
    CartesianIndex(15, 36, 3),
    CartesianIndex(27, 37, 5),
    CartesianIndex(33, 35, 7),
]

charges = NeutralChargeHopping(
    g;
    positive = vacancy_positions,
    negative = electron_positions,
    positive_effects = vacancy_effects,
    negative_effects = electron_effects,
    positive_charge = vacancy_charge,
    negative_charge = -electron_charge,
)

langevin_algorithm = LocalLangevin(stepsize = stepsize, adjusted = false)
vacancy_algorithm = Metropolis()

algorithm = @CompositeAlgorithm begin
    @alias langevin = langevin_algorithm
    @alias vacancy_metro = vacancy_algorithm

    @replace langevin.T => vacancy_metro.T

    langevin()
    @every defect_step_interval vacancy_metro()
end

graph_host = interface(g; title = "40x40x10 Coulomb spins with charged vacancies")
charge_host = interface(
    charges;
    title = "Positive vacancies (red) and mobile electrons (cyan)",
    positive_markersize = 0.62,
    negative_markersize = 0.36,
    positive_color = :red,
    negative_color = :cyan,
    lattice_color = (:gray70, 0.025),
)
process = createProcessManual(
    g,
    algorithm,
    StatefulAlgorithms.Init(:langevin; model = g),
    StatefulAlgorithms.Init(:vacancy_metro; model = charges),
    StatefulAlgorithms.Interactive(:langevin, :T),
)
