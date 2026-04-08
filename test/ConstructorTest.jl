using Test
using InteractiveIsing

const BETWEEN_LAYER_WG = @WG (;dr) -> 1.0f0 NN = 1

@testset "Multi-layer Constructor" begin
    l1 = Layer(2, Continuous(), Coords(0, 0, 0))
    l2 = Layer(2, Continuous(), Coords(0, 1, 0))

    g = IsingGraph(l1, BETWEEN_LAYER_WG, l2; precision = Float32)

    l1_idxs = collect(InteractiveIsing.graphidxs(g[1]))
    l2_idxs = collect(InteractiveIsing.graphidxs(g[2]))

    @test Matrix(adj(g)[l1_idxs, l1_idxs]) == zeros(Float32, 2, 2)
    @test Matrix(adj(g)[l2_idxs, l2_idxs]) == zeros(Float32, 2, 2)
    @test Matrix(adj(g)[l2_idxs, l1_idxs]) == ones(Float32, 2, 2)
    @test Matrix(adj(g)[l1_idxs, l2_idxs]) == ones(Float32, 2, 2)
end
