using Test
using InteractiveIsing
using SparseArrays

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

@testset "Constructor Adjacency Storage Type" begin
    l1 = Layer(2, Continuous(); periodic = false)
    l2 = Layer(2, Continuous(); periodic = false)

    sparse_graph = IsingGraph(l1, BETWEEN_LAYER_WG, l2; precision = Float32, adj = SparseMatrixCSC)
    @test adj(sparse_graph) isa SparseMatrixCSC{Float32,Int32}
    @test Matrix(adj(sparse_graph)) == [
        0f0 0f0 1f0 1f0
        0f0 0f0 1f0 1f0
        1f0 1f0 0f0 0f0
        1f0 1f0 0f0 0f0
    ]

    concrete_sparse_graph = IsingGraph(l1, BETWEEN_LAYER_WG, l2; precision = Float32, adj = SparseMatrixCSC{Float64,Int64})
    @test adj(concrete_sparse_graph) isa SparseMatrixCSC{Float64,Int64}

    regenerated = IsingGraph(l1, l2; precision = Float32, adj = SparseMatrixCSC)
    @test adj(regenerated) isa SparseMatrixCSC{Float32,Int32}
    genAdj!(regenerated[1], regenerated[2], BETWEEN_LAYER_WG)
    @test adj(regenerated) isa SparseMatrixCSC{Float32,Int32}
    @test Matrix(adj(regenerated)) == Matrix(adj(sparse_graph))
end

@testset "Initial State" begin
    bounded = IsingGraph(3, 3, Continuous(), StateSet(-1.0f0, 1.0f0); precision = Float32)
    @test all(x -> -1.0f0 <= x <= 1.0f0, InteractiveIsing.graphstate(bounded))

    explicit_scalar = IsingGraph(3, 3, Continuous(), StateSet(-Inf32, Inf32); precision = Float32, initial_state = 0.25f0)
    @test all(==(0.25f0), InteractiveIsing.graphstate(explicit_scalar))

    explicit_array = fill(0.5f0, 9)
    explicit_graph = IsingGraph(3, 3, Continuous(), StateSet(-Inf32, Inf32); precision = Float32, initial_state = explicit_array)
    @test InteractiveIsing.graphstate(explicit_graph) == explicit_array

    backing_a = fill(0.1f0, 4)
    backing_b = fill(0.2f0, 5)
    custom_state = RefConcatVector(backing_a, backing_b)
    custom_graph = IsingGraph(3, 3, Continuous(), StateSet(-Inf32, Inf32); precision = Float32, state = custom_state)
    @test InteractiveIsing.graphstate(custom_graph) === custom_state
    @test refslices(custom_state) === (backing_a, backing_b)
    @test sliceranges(custom_state) == (1:4, 5:9)
    @test collect(InteractiveIsing.graphstate(custom_graph)) == [fill(0.1f0, 4); fill(0.2f0, 5)]

    filled_state = RefConcatVector(fill(0.0f0, 4), fill(0.0f0, 5))
    IsingGraph(3, 3, Continuous(), StateSet(-Inf32, Inf32); precision = Float32, state = filled_state, initial_state = 0.75f0)
    @test all(==(0.75f0), filled_state)

    filled_state .= 2 .* explicit_array .+ 1
    @test collect(filled_state) == collect(2 .* explicit_array .+ 1)

    turbo_fill!(filled_state, 0.25f0)
    @test all(==(0.25f0), filled_state)

    turbo_copyto!(filled_state, explicit_array)
    @test collect(filled_state) == explicit_array

    turbo_map!(x -> x + 1.0f0, filled_state, explicit_array)
    @test collect(filled_state) == explicit_array .+ 1.0f0

    fallback = IsingGraph(3, 3, Continuous(), StateSet(-Inf32, Inf32), Ising(c = ConstVal(0f0), b = 0); precision = Float32)
    @test all(iszero, InteractiveIsing.graphstate(fallback))

    local_minimum = IsingGraph(
        3, 3,
        Continuous(),
        StateSet(-Inf32, Inf32),
        Ising(c = ConstVal(2f0), localpotential = ConstFill(1f0), b = ConstFill(4f0));
        precision = Float32,
    )
    @test all(x -> isapprox(x, 1.0f0; atol = 1f-6), InteractiveIsing.graphstate(local_minimum))

    quartic_double_well = IsingGraph(
        3, 3,
        Continuous(),
        StateSet(-Inf32, Inf32),
        Quartic(c = ConstVal(1f0), localpotential = ConstFill(1f0)) +
        Quadratic(c = ConstVal(-2f0), localpotential = ConstFill(1f0));
        precision = Float32,
    )
    @test all(x -> isapprox(abs(x), 1.0f0; atol = 1f-5), InteractiveIsing.graphstate(quartic_double_well))
end

@testset "Constructor errors" begin
    err = try
        IsingGraph(2, 2, Continuous(), Quadratic(), Quartic(); precision = Float32)
        nothing
    catch e
        e
    end

    @test err isa ArgumentError
    @test occursin("single Hamiltonian argument", sprint(showerror, err))
    @test occursin("accidental comma", sprint(showerror, err))
end
