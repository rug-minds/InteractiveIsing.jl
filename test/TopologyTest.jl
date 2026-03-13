using Test
using InteractiveIsing

@testset "Topology Distances" begin
    g = IsingGraph(10, 10; periodic = true, type = Continuous)
    top = topology(g[1])

    @test InteractiveIsing.dist(top, CartesianIndex(1, 1), CartesianIndex(10, 1)) == 1.0
    @test InteractiveIsing.dist(top, CartesianIndex(1, 1), CartesianIndex(6, 1)) == 5.0

    setdist!(top, (2.0, 3.0))
    @test InteractiveIsing.dist(top, CartesianIndex(1, 1), CartesianIndex(10, 1)) == 2.0
    @test InteractiveIsing.dist(top, CartesianIndex(1, 1), CartesianIndex(1, 10)) == 3.0

    g_np = IsingGraph(10, 10; periodic = false, type = Continuous)
    top_np = topology(g_np[1])

    @test InteractiveIsing.dist(top_np, CartesianIndex(1, 1), CartesianIndex(10, 1)) == 9.0
    @test InteractiveIsing.dist(top_np, CartesianIndex(1, 1), CartesianIndex(1, 10)) == 9.0
end
