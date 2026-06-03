using Test
using InteractiveIsing
using InteractiveIsing.Processes
using LinearAlgebra
using SparseArrays
using StaticArrays

@testset "VectorSpinGraph construction and Hamiltonians" begin
    adj = InteractiveIsing.UndirectedAdjacency(
        sparse([1, 2], [2, 1], Float32[1, 1], 2, 2),
        zeros(Float32, 2),
    )
    initial_state = SVector{3,Float32}[
        SVector{3,Float32}(1, 0, 0),
        SVector{3,Float32}(0, 1, 0),
    ]

    g = VectorSpinGraph(
        2,
        Continuous(),
        StateSet(-1, 1),
        VectorSpin(h = SVector{3,Float32}(0, 0, 1));
        dimension = 3,
        precision = Float32,
        adj,
        initial_state,
    )

    @test g isa InteractiveIsing.AbstractVectorSpinGraph{Float32,3}
    @test eltype(g) === Float32
    @test eltype(InteractiveIsing.graphstate(g)) === SVector{3,Float32}
    @test InteractiveIsing.spin_dimension(g) == 3
    @test all(s -> isapprox(norm(s), 1f0; atol = 1f-6), InteractiveIsing.state(g))

    @test InteractiveIsing.calculate(InteractiveIsing.H(), g.hamiltonian, g) ≈ 0f0

    proposal = FlipProposal(
        1,
        initial_state[1],
        SVector{3,Float32}(0, 1, 0),
        1,
    )
    @test InteractiveIsing.calculate(InteractiveIsing.ΔH(), g.hamiltonian, g, proposal) ≈ -1f0
    @test InteractiveIsing.calculate(InteractiveIsing.d_iH(), g.hamiltonian, g, 1) ≈ SVector{3,Float32}(0, -1, -1)

    random_proposal = InteractiveIsing.random_proposal(g)
    @test random_proposal isa FlipProposal{SVector{3,Float32}}
    @test isapprox(norm(InteractiveIsing.to_val(random_proposal)), 1f0; atol = 1f-6)
end

@testset "VectorSpinGraph bounded component mode" begin
    initial_state = SVector{2,Float32}[
        SVector{2,Float32}(0.25, 0.5),
        SVector{2,Float32}(-0.75, 0.125),
    ]
    g = VectorSpinGraph(
        2,
        Continuous(),
        StateSet(-1, 1),
        VectorSpin(),
        VectorSpinLocalProposer(0f0);
        dimension = 2,
        precision = Float32,
        initial_state,
        unit_norm = false,
    )

    @test !InteractiveIsing.vector_unit_norm(g)
    @test any(s -> !isapprox(norm(s), 1f0; atol = 1f-6), InteractiveIsing.state(g))

    local_proposal = rand(InteractiveIsing.get_proposer(g))
    @test InteractiveIsing.to_val(local_proposal) == InteractiveIsing.from_val(local_proposal)
    @test !isapprox(norm(InteractiveIsing.to_val(local_proposal)), 1f0; atol = 1f-6)

    random_graph = VectorSpinGraph(
        16,
        Continuous(),
        StateSet(-1, 1),
        VectorSpin(),
        VectorSpinProposer();
        dimension = 2,
        precision = Float32,
        unit_norm = false,
    )
    random_proposal = rand(InteractiveIsing.get_proposer(random_graph))
    @test all(component -> -1f0 <= component <= 1f0, InteractiveIsing.to_val(random_proposal))
    @test !isapprox(norm(InteractiveIsing.to_val(random_proposal)), 1f0; atol = 1f-6)
end

@testset "VectorMagnitudePenalty Hamiltonian" begin
    initial_state = SVector{2,Float32}[
        SVector{2,Float32}(1, 0),
        SVector{2,Float32}(0.5, 0),
    ]
    g = VectorSpinGraph(
        2,
        Continuous(),
        StateSet(-1, 1),
        VectorMagnitudePenalty(c = 2f0, target = 1f0);
        dimension = 2,
        precision = Float32,
        initial_state,
        unit_norm = false,
    )

    @test InteractiveIsing.calculate(InteractiveIsing.H(), g.hamiltonian, g) ≈ 0.5f0

    proposal = FlipProposal(2, initial_state[2], SVector{2,Float32}(1, 0), 1)
    @test InteractiveIsing.calculate(InteractiveIsing.ΔH(), g.hamiltonian, g, proposal) ≈ -0.5f0
    @test InteractiveIsing.calculate(InteractiveIsing.d_iH(), g.hamiltonian, g, 2) ≈ SVector{2,Float32}(-2, 0)
end

@testset "VectorSpinGraph Metropolis process compatibility" begin
    adj = InteractiveIsing.UndirectedAdjacency(
        sparse([1, 2], [2, 1], Float32[1, 1], 2, 2),
        zeros(Float32, 2),
    )
    g = VectorSpinGraph(2, Continuous(), StateSet(-1, 1), VectorSpin(); dimension = 3, precision = Float32, adj)
    before = copy(InteractiveIsing.state(g))

    process = createProcess(g, Metropolis(); repeats = 8)
    wait(process)

    @test length(InteractiveIsing.state(g)) == 2
    @test all(s -> isapprox(norm(s), 1f0; atol = 1f-6), InteractiveIsing.state(g))
    @test InteractiveIsing.state(g) != before
end

@testset "VectorSpinGraph interactive temperature process input" begin
    g = VectorSpinGraph(
        4,
        Continuous(),
        StateSet(-1, 1),
        VectorSpin();
        dimension = 2,
        precision = Float32,
        unit_norm = false,
    )
    temp!(g, 0.35f0)
    g.addons[:interactive] = true

    process = createProcess(g, Metropolis(); lifetime = 1)
    wait(process)
    subcontexts = Processes.get_subcontexts(getcontext(process))
    key = only(filter(name -> name !== :globals, propertynames(subcontexts)))
    T = getproperty(Processes.getdata(getproperty(subcontexts, key)), :T)

    @test T isa Processes.InteractiveVar{Float32}
    @test T[] ≈ 0.35f0
end

@testset "VectorSpinGraph LocalLangevin dynamics keyword" begin
    g = VectorSpinGraph(
        4,
        Continuous(),
        StateSet(-1, 1),
        VectorSpin() + VectorMagnitudePenalty(c = 2f0, target = 0.6f0);
        dimension = 2,
        precision = Float32,
        unit_norm = false,
    )

    process = createProcess(
        g;
        dynamics = LocalLangevin(stepsize = 0.0f0, adjusted = false, order = :deterministic),
        lifetime = 1,
    )
    wait(process)
    subcontexts = Processes.get_subcontexts(getcontext(process))
    key = only(filter(name -> name !== :globals, propertynames(subcontexts)))

    @test key === :LocalLangevin_1
    @test getproperty(getproperty(Processes.getcontext(process), key), :model) === g

    adjusted_process = createProcess(
        g;
        dynamics = LocalLangevin(stepsize = 0.0f0),
        lifetime = 1,
    )
    wait(adjusted_process)
    adjusted_subcontexts = Processes.get_subcontexts(getcontext(adjusted_process))
    adjusted_key = only(filter(name -> name !== :globals, propertynames(adjusted_subcontexts)))

    @test adjusted_key === :LocalLangevin_1
end

@testset "VectorSpinGraph 3D vector spins on 3D graph" begin
    g = VectorSpinGraph(
        3,
        3,
        2,
        Continuous(),
        StateSet(-1, 1),
        VectorSpin() + VectorMagnitudePenalty(c = 2f0, target = 0.6f0);
        dimension = 3,
        precision = Float32,
        unit_norm = false,
    )

    @test InteractiveIsing.spin_dimension(g) == 3
    @test size(g[1]) == (3, 3, 2)
    @test eltype(InteractiveIsing.state(g)) === SVector{3,Float32}

    process = createProcess(g, dynamics = LocalLangevin(stepsize = 0.01f0), lifetime = 1)
    wait(process)
    subcontexts = Processes.get_subcontexts(getcontext(process))
    key = only(filter(name -> name !== :globals, propertynames(subcontexts)))
    data = Processes.getdata(getproperty(subcontexts, key))

    @test key === :LocalLangevin_1
    @test data.dH_prealloc isa Vector{SVector{3,Float32}}
end

@testset "VectorSpinGraph rejects scalar Ising Hamiltonian" begin
    @test_throws ArgumentError VectorSpinGraph(
        2,
        Continuous(),
        StateSet(-1, 1),
        Ising();
        dimension = 3,
    )
end
