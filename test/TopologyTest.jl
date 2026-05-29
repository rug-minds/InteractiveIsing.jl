using Test
using InteractiveIsing
using SparseArrays
using LinearAlgebra

hex_nearest_weight(; dr) = dr == 1 ? 1.0f0 : 0.0f0
const HEX_NEAREST_WG = @WG hex_nearest_weight NN = 1

zigzag_nearest_weight(; dr) = isapprox(dr, 1.0f0; atol = 1.0f-6) ? 1.0f0 : 0.0f0
const ZIGZAG_NEAREST_WG = @WG zigzag_nearest_weight NN = 1

@testset "Topology Distances" begin
    g = IsingGraph(10, 10, Continuous(); periodic = true)
    top = topology(g[1])

    @test InteractiveIsing.dist(top, CartesianIndex(1, 1), CartesianIndex(10, 1)) == 1.0
    @test InteractiveIsing.dist(top, CartesianIndex(1, 1), CartesianIndex(6, 1)) == 5.0

    setdist!(top, (2.0, 3.0))
    @test InteractiveIsing.dist(top, CartesianIndex(1, 1), CartesianIndex(10, 1)) == 2.0
    @test InteractiveIsing.dist(top, CartesianIndex(1, 1), CartesianIndex(1, 10)) == 3.0

    g_np = IsingGraph(10, 10, Continuous(); periodic = false)
    top_np = topology(g_np[1])

    @test InteractiveIsing.dist(top_np, CartesianIndex(1, 1), CartesianIndex(10, 1)) == 9.0
    @test InteractiveIsing.dist(top_np, CartesianIndex(1, 1), CartesianIndex(1, 10)) == 9.0

    square_template = SquareTopology(; lattice_constants = (2.0, 3.0), periodic = false)
    @test size(square_template) == (0, 0)
    square_sized = sizeto(square_template, (10, 10))
    @test size(square_sized) == (10, 10)
    @test InteractiveIsing.dist(square_sized, CartesianIndex(1, 1), CartesianIndex(2, 1)) == 2.0
end

@testset "Hexagonal Topology Distances" begin
    top = HexagonalTopology((10, 10); periodic = false)

    @test InteractiveIsing.dist(top, CartesianIndex(2, 2), CartesianIndex(3, 2)) == 1.0
    @test InteractiveIsing.dist(top, CartesianIndex(2, 2), CartesianIndex(2, 3)) == 1.0
    @test InteractiveIsing.dist(top, CartesianIndex(2, 2), CartesianIndex(3, 1)) == 1.0
    @test InteractiveIsing.dist(top, CartesianIndex(2, 2), CartesianIndex(3, 3)) ≈ sqrt(3.0)

    setdist!(top, (2.0, 2.0))
    @test InteractiveIsing.dist(top, CartesianIndex(2, 2), CartesianIndex(3, 2)) == 2.0
    @test InteractiveIsing.dist(top, CartesianIndex(2, 2), CartesianIndex(3, 1)) == 2.0

    periodic_top = HexagonalTopology((10, 10); periodic = true)
    @test InteractiveIsing.dist(periodic_top, CartesianIndex(1, 1), CartesianIndex(10, 1)) == 1.0
    @test InteractiveIsing.dist(periodic_top, CartesianIndex(1, 1), CartesianIndex(1, 10)) == 1.0
    @test InteractiveIsing.dist(periodic_top, CartesianIndex(1, 1), CartesianIndex(10, 2)) == 1.0

    hex_template = HexagonalTopology(; periodic = false)
    @test size(hex_template) == (0, 0)
    hex_sized = sizeto(hex_template, (10, 10))
    @test size(hex_sized) == (10, 10)
    @test InteractiveIsing.dist(hex_sized, CartesianIndex(2, 2), CartesianIndex(3, 1)) == 1.0
end

@testset "Hexagonal Topology Connections" begin
    top = HexagonalTopology((3, 3); periodic = false)
    g = IsingGraph(3, 3, Continuous(), HEX_NEAREST_WG, top; periodic = false)
    adjmat = Matrix(sparse(adj(g)))

    center_idx = LinearIndices((3, 3))[CartesianIndex(2, 2)]
    corner_idx = LinearIndices((3, 3))[CartesianIndex(1, 1)]

    @test count(!iszero, adjmat[:, center_idx]) == 6
    @test count(!iszero, adjmat[:, corner_idx]) == 2
end

@testset "General Lattice Topology" begin
    top = LatticeTopology(
        (4, 4),
        (2.0, 0.0),
        (0.0, 3.0);
        periodic = false,
        lattice_type = Rectangular,
    )

    @test latticetype(top) === Rectangular
    @test InteractiveIsing.dist(top, CartesianIndex(1, 1), CartesianIndex(2, 1)) == 2.0
    @test InteractiveIsing.dist(top, CartesianIndex(1, 1), CartesianIndex(1, 2)) == 3.0
    @test dot(covectors(top)[1], primitive_vectors(top)[1]) ≈ 1.0
    @test dot(covectors(top)[1], primitive_vectors(top)[2]) ≈ 0.0

    int_vector_template = LatticeTopology((2, 0), (0, 3); periodic = false)
    @test size(int_vector_template) == (0, 0)
    @test InteractiveIsing.lattice_constants(sizeto(int_vector_template, (4, 4))) == (2.0f0, 3.0f0)

    top3 = LatticeTopology(
        (1.0f0, 0.0f0, 0.0f0),
        (0.0f0, 2.0f0, 0.0f0),
        (0.0f0, 0.0f0, 3.0f0);
        periodic = false,
    )

    @test size(top3) == (0, 0, 0)
    sized_top3 = sizeto(top3, (2, 2, 2))
    @test size(sized_top3) == (2, 2, 2)
    @test InteractiveIsing.dist(sized_top3, CartesianIndex(1, 1, 1), CartesianIndex(1, 1, 2)) == 3.0f0
end

@testset "ZigZag Lattice Topology" begin
    row_spacing = sqrt(3.0f0) / 2
    top = LatticeTopology(
        (0.0f0, row_spacing),
        (1.0f0, 0.0f0);
        layout = ZigZagRows(),
        periodic = false,
        lattice_type = Hexagonal,
    )
    @test size(top) == (0, 0)

    sized_top = sizeto(top, (3, 3))

    @test !InteractiveIsing.is_translation_invariant(sized_top)
    @test InteractiveIsing.lattice_constants(sized_top)[1] == row_spacing
    @test InteractiveIsing.dist(sized_top, CartesianIndex(2, 2), CartesianIndex(1, 2)) ≈ 1.0f0
    @test InteractiveIsing.dist(sized_top, CartesianIndex(2, 2), CartesianIndex(1, 1)) ≈ sqrt(3.0f0)

    g = IsingGraph(3, 3, Continuous(), ZIGZAG_NEAREST_WG, top; periodic = false)
    @test size(topology(g[1])) == (3, 3)
    adjmat = Matrix(sparse(adj(g)))
    center_idx = LinearIndices((3, 3))[CartesianIndex(2, 2)]
    corner_idx = LinearIndices((3, 3))[CartesianIndex(1, 1)]

    @test count(!iszero, adjmat[:, center_idx]) == 6
    @test count(!iszero, adjmat[:, corner_idx]) == 2
end
