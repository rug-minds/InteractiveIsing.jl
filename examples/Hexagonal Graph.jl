using InteractiveIsing
using InteractiveIsing.StatefulAlgorithms

"""
    zigzag_hexagonal_isingweights(; dr)

Return the nearest-neighbor Ising coupling for a zigzag hexagonal lattice.
"""
function zigzag_hexagonal_isingweights(; dr::R) where {R}
    return isapprox(dr, one(R); atol = R(1e-6)) ? 1f0 : 0f0
end

# Row-zigzag layout gives the finite patch square-like bounds with staggered
# nearest-neighbor rows instead of the axial-coordinate parallelogram.
row_spacing = sqrt(3f0) / 2
top = LatticeTopology(
    (0f0, row_spacing, 0f0),
    (1f0, 0f0, 0f0),
    (0f0, 0f0, 1f0);
    layout = ZigZagRows(),
    periodic = true,
    lattice_type = Hexagonal,
)
wg = @WG zigzag_hexagonal_isingweights NN = 1

g = IsingGraph(
    40,
    40,
    10,
    Continuous(),
    LocalProposer(0.5f0),
    wg,
    top,
    StateSet(-1f0, 1f0),
    Ising(c = ConstVal(0f0), b = 0f0, localpotential = 0f0);
    periodic = true
)

host = interface(g)
createProcess(g)
