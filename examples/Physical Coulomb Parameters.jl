using InteractiveIsing
using Unitful

# CoulombHamiltonian is a 3D layer term. This example keeps all physical
# conversion at construction time while the instantiated graph stores plain
# Float32 values.
dipole_scale = 3.33564e-30u"C*m" # one Debye in SI units
scales = PhysicalScales(
    energy = 1u"meV",
    dipole = dipole_scale,
)

temperature = 45u"K"
kBT = Unitful.uconvert(u"meV", Unitful.k * temperature)

"""
    nearest_neighbor_exchange(; dr, c1 = nothing, c2 = nothing, dc = nothing)

Return a Unitful exchange coupling for the generated adjacency. The physical
wrapper supplies `dr` as a Unitful 3D lattice distance.
"""
function nearest_neighbor_exchange(; dr, c1 = nothing, c2 = nothing, dc = nothing)
    amplitude = 3.0u"meV"
    cutoff = 0.55u"nm"
    decay_length = 0.35u"nm"

    x = Unitful.ustrip(Unitful.NoUnits, dr / decay_length)
    return dr <= cutoff ? amplitude * exp(-x) : 0.0u"meV"
end

wg = PhysicalWeightGenerator(WeightGenerator(nearest_neighbor_exchange, 1))

hamiltonian =
    Ising(c = ConstVal(0.0), b = 0.15u"meV") +
    CoulombHamiltonian(
        scaling = 1.5 * dipole_scale,
        screening = 2.0u"nm",
        recalc = 20,
    )

g = IsingGraph(
    6,
    6,
    3,
    Continuous(),
    LatticeConstants(0.5u"nm", 0.5u"nm", 0.75u"nm"),
    wg,
    StateSet(-1.0f0, 1.0f0),
    hamiltonian;
    periodic = (:x, :y),
    precision = Float32,
    physical_scales = scales,
    temperature = kBT,
)

@assert physicalscales(g).length[] == 1u"nm"
@assert temp(g) ≈ Float32(Unitful.ustrip(kBT))

mag = InteractiveIsing.gethamiltonian(g.hamiltonian, InteractiveIsing.ExtField)
coulomb = InteractiveIsing.gethamiltonian(g.hamiltonian, InteractiveIsing.CoulombHamiltonian)

@assert mag.b[1] == 0.15f0
@assert coulomb.scaling[] ≈ 1.5f0
@assert coulomb.screen_top ≈ 2.0f0
@assert coulomb.screen_bot ≈ 2.0f0

println("grid: ", size(g))
println("length scale: ", physicalscales(g).length[])
println("temperature: ", temperature, " (k_B T = ", temp(g), " meV)")
println("external field: ", physicalvalue(g, InteractiveIsing.ExtField, :b)[1])
println("coulomb dipole scale: ", coulomb.scaling[] * physicalscales(g).dipole[])
println("screening length: ", coulomb.screen_top * physicalscales(g).length[])
println("strongest exchange coupling: ", maximum(abs, Matrix(adj(g))), " meV")
