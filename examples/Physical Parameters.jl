using InteractiveIsing
using Unitful

# This example uses physical inputs in meV, nm, and kelvin. The Monte Carlo
# runtime temperature is still k_B T in meV, so kelvin is converted explicitly
# with Unitful.k at the graph boundary.
scales = PhysicalScales(
    energy = 1u"meV",
)

initial_temperature = 25u"K"
warmer_temperature = 35u"K"
initial_kBT = Unitful.uconvert(u"meV", Unitful.k * initial_temperature)
warmer_kBT = Unitful.uconvert(u"meV", Unitful.k * warmer_temperature)

"""
    exchange_weight(; dr, c1 = nothing, c2 = nothing, dc = nothing)

Return a Unitful nearest-neighbor exchange coupling. `dr` is Unitful because
the generator is wrapped in `PhysicalWeightGenerator` below.
"""
function exchange_weight(; dr, c1 = nothing, c2 = nothing, dc = nothing)
    decay_length = 0.35u"nm"
    cutoff = 0.55u"nm"
    amplitude = 8.0u"meV"

    x = Unitful.ustrip(Unitful.NoUnits, dr / decay_length)
    return dr <= cutoff ? amplitude * exp(-x) : 0.0u"meV"
end

wg = PhysicalWeightGenerator(WeightGenerator(exchange_weight, 1))

g = IsingGraph(
    8,
    8,
    4,
    Continuous(),
    LatticeConstants(0.5u"nm", 0.5u"nm", 0.5u"nm"),
    wg,
    StateSet(-1.0f0, 1.0f0),
    Ising(c = ConstVal(0.0), b = 0.25u"meV");
    periodic = (:x, :y),
    precision = Float32,
    physical_scales = scales,
    temperature = initial_kBT,
)

# The Unitful lattice constants inferred the shared length scale for `dr`.
@assert physicalscales(g).length[] == 1u"nm"

# Existing runtime storage is unit-free.
mag = InteractiveIsing.gethamiltonian(g.hamiltonian, InteractiveIsing.MagField)
@assert mag.b[1] == 0.25f0
@assert temp(g) ≈ Float32(Unitful.ustrip(initial_kBT))

# Explicit physical setters convert once and write internal numeric values.
setphysical!(g, InteractiveIsing.MagField, :b, 0.4u"meV")
temp!(g, warmer_kBT)

field = physicalvalue(g, InteractiveIsing.MagField, :b)
adjacency = Matrix(adj(g))

println("length scale: ", physicalscales(g).length[])
println("grid: ", size(g))
println("temperature: ", warmer_temperature, " (k_B T = ", temp(g), " meV)")
println("field: ", field[1])
println("strongest generated coupling: ", maximum(abs, adjacency), " meV")
run_interface(g)