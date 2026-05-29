using Test
using InteractiveIsing
using SparseArrays

@testset "UndirectedAdjacency Reconstruction" begin
    rows = Int32[2, 1, 3, 2]
    cols = Int32[1, 2, 2, 3]
    vals = Float32[1, 1, 2, 2]

    sp = sparse(rows, cols, vals, 3, 3)
    diag = Float32[10, 20, 30]
    base = InteractiveIsing.UndirectedAdjacency(sp, diag; fastwrite = true)

    replacement_sp = sparse(rows, cols, Float32[5, 5, 7, 7], 3, 3)
    replacement_diag = Float32[3, 4, 5]

    replaced_sp = InteractiveIsing.UndirectedAdjacency(base, replacement_sp)
    @test sparse(replaced_sp) == replacement_sp
    @test replaced_sp.diag == diag
    @test InteractiveIsing.fastwrite(replaced_sp)
    replaced_sp[1, 2] = 9f0
    @test replaced_sp[1, 2] == 9f0
    @test replaced_sp[2, 1] == 9f0

    replaced_diag = InteractiveIsing.instantiate(base; diag = replacement_diag)
    @test sparse(replaced_diag) == sparse(base)
    @test replaced_diag.diag == replacement_diag

    replaced_both = InteractiveIsing.UndirectedAdjacency(base, replacement_sp; diag = replacement_diag)
    @test sparse(replaced_both) == replacement_sp
    @test replaced_both.diag == replacement_diag

    wrong_topology = sparse(Int32[1, 3], Int32[3, 1], Float32[1, 1], 3, 3)
    @test_throws ArgumentError InteractiveIsing.UndirectedAdjacency(base, wrong_topology)

    wrong_diag = Float32[1, 2]
    @test_throws DimensionMismatch InteractiveIsing.instantiate(base; diag = wrong_diag)
end

@testset "SparseMatrixCSC Neighbor Loops" begin
    rows = Int32[1, 2, 3, 2]
    cols = Int32[2, 2, 2, 3]
    vals = Float32[10, 20, 30, 40]
    sp = sparse(rows, cols, vals, 3, 3)
    nodevals = Float32[2, 3, 5]

    @test InteractiveIsing.weighted_neighbors_sum(2, sp, nodevals) == 10f0 * 2f0 + 30f0 * 5f0
    @test InteractiveIsing.weighted_self(2, sp, nodevals) == 20f0 * 3f0
    @test collect(InteractiveIsing.index_pairs_iterator(sp)) == [(1, 2), (3, 2), (2, 3)]
    @test collect(InteractiveIsing.connection_iterator(sp, true)) ==
          [(1, 2, 10f0), (2, 2, 20f0), (3, 2, 30f0), (2, 3, 40f0)]

    transformed = InteractiveIsing.weighted_neighbors_sum(
        2,
        sp,
        nodevals;
        transform = x -> x + 1f0,
        transform_weight = w -> 2f0 * w,
    )
    @test transformed == 2f0 * 10f0 * 3f0 + 2f0 * 30f0 * 6f0
end
