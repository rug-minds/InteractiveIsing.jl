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

    replaced_diag = InteractiveIsing.reconstruct(base; diag = replacement_diag)
    @test sparse(replaced_diag) == sparse(base)
    @test replaced_diag.diag == replacement_diag

    replaced_both = InteractiveIsing.UndirectedAdjacency(base, replacement_sp; diag = replacement_diag)
    @test sparse(replaced_both) == replacement_sp
    @test replaced_both.diag == replacement_diag

    wrong_topology = sparse(Int32[1, 3], Int32[3, 1], Float32[1, 1], 3, 3)
    @test_throws ArgumentError InteractiveIsing.UndirectedAdjacency(base, wrong_topology)

    wrong_diag = Float32[1, 2]
    @test_throws DimensionMismatch InteractiveIsing.reconstruct(base; diag = wrong_diag)
end
