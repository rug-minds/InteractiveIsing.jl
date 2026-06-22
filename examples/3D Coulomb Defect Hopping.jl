using InteractiveIsing
using InteractiveIsing.StatefulAlgorithms
using Random
using Unitful

"""
    coulomb_defect_weights(; dr)

Nearest-neighbor coupling used by the bilinear part of the 3D graph. `dr` is
Unitful because this example keeps the model parameters in meV/nm units.
"""
function coulomb_defect_weights(; dr)
    distance_nm = Unitful.ustrip(u"nm", dr)
    return distance_nm == 1 ? 1f0u"meV" : 0f0u"meV"
end

nx, ny, nz = 40, 40, 10
wg = PhysicalWeightGenerator(WeightGenerator(coulomb_defect_weights, 1))
elementary_charge = 1.602176634f-19u"C"

# Keep the internal Hamiltonian scale in meV. Using eV here moves the same
# dimensionless transition scale up by a factor of 1000 in the Kelvin readout.
scales = PhysicalScales(
    energy = 1u"meV",
    length = 1u"nm",
    charge = elementary_charge,
    dipole = elementary_charge * 1u"nm",
)

vacancy_stiffness = zeros(Float32, nx * ny * nz)
vacancy_quartic = zeros(Float32, nx * ny * nz)

stepsize = 0.025f0
temperature = 45u"K"
kBT = Unitful.uconvert(u"meV", Unitful.k * temperature)
defect_step_interval = 1000

coulomb_scaling = elementary_charge * 1f0u"nm"
coulomb_screening = 8f0u"nm"
coulomb_recalc_interval = 1001
electron_charge = elementary_charge
vacancy_charge = 2f0 * electron_charge
external_field_z = 6f0u"meV"
electron_attempt_rate = 10f0

vacancy_hamiltonian =
    CoulombChargeCoupling(vacancy_charge; split = 0.5f0) +
    ExternalFieldChargeCoupling() +
    LocalPotentialShiftCoupling(2, 0.012f0) +
    LocalPotentialShiftCoupling(4, 0.004f0)

electron_hamiltonian =
    CoulombChargeCoupling(-electron_charge; split = 0.5f0) +
    ExternalFieldChargeCoupling()

g = IsingGraph(
    nx,
    ny,
    nz,
    Continuous(),
    wg,
    LatticeConstants(1f0u"nm", 1f0u"nm", 1f0u"nm"),
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

temp!(g, kBT)

charges = DefectsModel(
    g;
    vacancies = MobileVacancies(10; charge = vacancy_charge, hamiltonian = vacancy_hamiltonian),
    charges = MobileCharges(20; charge = -electron_charge, hamiltonian = electron_hamiltonian),
    electron_attempt_rate,
    rng = MersenneTwister(42),
)

println("Field drift demo: positive vacancies should drift toward z = $nz; electrons should drift toward z = 1.")

langevin_algorithm = LocalLangevin(stepsize = stepsize, adjusted = false)
charge_algorithm = Metropolis()

dynamics = @CompositeAlgorithm begin
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
    dynamics,
    StatefulAlgorithms.Init(:langevin; model = g),
    StatefulAlgorithms.Init(:charge_metro; model = charges),
    StatefulAlgorithms.Interactive(:langevin, :T),
)
