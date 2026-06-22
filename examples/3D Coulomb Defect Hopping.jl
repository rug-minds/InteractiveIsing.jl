using InteractiveIsing
using InteractiveIsing.StatefulAlgorithms
using Unitful

"""
    coulomb_defect_weights(; dr)

Nearest-neighbor coupling used by the bilinear part of the 3D graph.
"""
coulomb_defect_weights(; dr) = dr == 1 ? 1f0 : 0f0

nx, ny, nz = 40, 40, 10
wg = @WG coulomb_defect_weights NN = 1
scales = PhysicalScales(energy = 1u"eV", charge = 1u"C")

vacancy_stiffness = zeros(Float32, nx * ny * nz)
vacancy_quartic = zeros(Float32, nx * ny * nz)

stepsize = 0.025f0
temperature = 0.04f0
defect_step_interval = 1000

coulomb_scaling = 1f0
coulomb_screening = 8f0
coulomb_recalc_interval = Inf
electron_charge = 1f0u"C"
vacancy_charge = 2f0 * electron_charge
external_field_z = 6f0u"eV"
vacancy_attempt_rate = 1f0
electron_attempt_rate = 10f0

charged_oxygen_vacancy = (
    CoulombChargeShift(vacancy_charge; split = 0.5f0),
    ExtFieldChargeCoupling(),
    LocalPotentialShift(2, 0.012f0),
    LocalPotentialShift(4, 0.004f0),
)

mobile_electron = (
    CoulombChargeShift(-electron_charge; split = 0.5f0),
    ExtFieldChargeCoupling(),
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
        ) + ExtField(b = external_field_z),
    periodic = (:x, :y),
    precision = Float32,
    initial_state = 0f0,
    physical_scales = scales,
)

temp!(g, temperature)

vacancy_positions = [
    CartesianIndex(6, 6, 5),
    CartesianIndex(14, 10, 5),
    CartesianIndex(22, 8, 5),
    CartesianIndex(32, 12, 5),
    CartesianIndex(9, 24, 5),
    CartesianIndex(18, 28, 5),
    CartesianIndex(27, 30, 6),
    CartesianIndex(35, 25, 6),
    CartesianIndex(11, 36, 6),
    CartesianIndex(24, 35, 6),
]

electron_positions = [
    CartesianIndex(4, 5, 5),
    CartesianIndex(8, 7, 5),
    CartesianIndex(12, 9, 5),
    CartesianIndex(16, 11, 5),
    CartesianIndex(20, 7, 6),
    CartesianIndex(24, 9, 7),
    CartesianIndex(28, 11, 6),
    CartesianIndex(34, 13, 6),
    CartesianIndex(7, 22, 5),
    CartesianIndex(11, 26, 5),
    CartesianIndex(15, 30, 5),
    CartesianIndex(19, 34, 6),
    CartesianIndex(23, 28, 7),
    CartesianIndex(29, 32, 6),
    CartesianIndex(33, 26, 6),
    CartesianIndex(37, 24, 6),
    CartesianIndex(9, 38, 5),
    CartesianIndex(15, 36, 5),
    CartesianIndex(27, 37, 6),
    CartesianIndex(33, 35, 6),
]

charges = ChargeHopProposer(
    g;
    positive = vacancy_positions,
    negative = electron_positions,
    positive_effects = vacancy_effects,
    negative_effects = electron_effects,
    positive_charge = vacancy_charge,
    negative_charge = -electron_charge,
    positive_attempt_rate = vacancy_attempt_rate,
    negative_attempt_rate = electron_attempt_rate,
)

println("Field drift demo: positive vacancies should drift toward z = $nz; electrons should drift toward z = 1.")

langevin_algorithm = LocalLangevin(stepsize = stepsize, adjusted = false)
charge_algorithm = Metropolis()

algorithm = @CompositeAlgorithm begin
    @alias langevin = langevin_algorithm
    @alias charge_metro = charge_algorithm

    @replace langevin.T => charge_metro.T

    langevin()
    @every defect_step_interval charge_metro()
end

graph_host = interface(g; title = "40x40x10 Coulomb spins with charged vacancies")
charge_host = interface(
    charges;
    title = "Positive vacancies (red) and mobile electrons (cyan)",
    positive_markersize = 0.62,
    negative_markersize = 0.36,
    positive_color = :red,
    negative_color = :cyan,
)
process = createProcessManual(
    g,
    algorithm,
    StatefulAlgorithms.Init(:langevin; model = g),
    StatefulAlgorithms.Init(:charge_metro; model = charges),
    StatefulAlgorithms.Interactive(:langevin, :T),
)
