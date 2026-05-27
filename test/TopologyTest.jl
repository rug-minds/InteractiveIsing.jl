using Test
using InteractiveIsing
using SparseArrays

hex_nearest_weight(; dr) = dr == 1 ? 1.0f0 : 0.0f0
const HEX_NEAREST_WG = @WG hex_nearest_weight NN = 1

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
