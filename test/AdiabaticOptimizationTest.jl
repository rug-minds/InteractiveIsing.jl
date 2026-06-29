# AI Generated

using Test
using InteractiveIsing

@testset "Whole-state derivative CPU fallback" begin
    hamiltonian = Ising(
        adj = Float32[0 1; 1 0],
        b = Float32[0.25, -0.5],
    ) + SoftplusMarginNudging(0.75f0, Float32[1, -1], Float32[1, 1], 0.4f0)
    graph = IsingGraph(
        2,
        Continuous(),
        StateSet(-1.0f0, 1.0f0),
        hamiltonian;
        precision = Float32,
        initial_state = Float32[0.25, -0.75],
    )
    initialized = InteractiveIsing.init!(graph.hamiltonian, graph)

    whole_state = InteractiveIsing.calculate(d_sH(), initialized, graph)
    scalar_state = similar(whole_state)
    spins = InteractiveIsing.graphstate(graph)
    for layer in InteractiveIsing.layers(graph)
        for spin_idx in InteractiveIsing.graphidxs(layer)
            proposal = SingleSpinProposal{eltype(graph)}(
                spin_idx,
                spins[spin_idx],
                NoChange(),
                InteractiveIsing.layeridx(layer),
                false,
            )
            scalar_state[spin_idx] = InteractiveIsing.calculate(InteractiveIsing.d_iH(), initialized, graph, proposal)
        end
    end

    @test whole_state ≈ scalar_state

    buffered_state = Float32[1.0, -1.0]
    buffered_derivative = similar(whole_state)
    InteractiveIsing.calculate!(
        buffered_derivative,
        d_sH(),
        initialized,
        graph;
        backend = CPUBackend(buffered_state),
    )
    @test InteractiveIsing.graphstate(graph) == Float32[0.25, -0.75]
    @test buffered_derivative != whole_state
end

@testset "Adiabatic dynamics smoke" begin
    x = Float32[-0.5, 0.25]
    p = Float32[0.1, -0.2]
    grad = Float32[0.3, -0.4]

    InteractiveIsing.run_adiabatic_dynamics!(x, p, grad, 1.2f0, 0.1f0, 0.05f0)

    @test length(x) == 2
    @test length(p) == 2
    @test all(abs.(x) .<= 1.0f0)
end

@testset "GPU derivative unsupported terms error" begin
    graph = IsingGraph(
        2,
        Continuous(),
        StateSet(-1.0f0, 1.0f0);
        precision = Float32,
        initial_state = Float32[0.1, -0.2],
    )
    backend = MetalBackend(InteractiveIsing, copy(InteractiveIsing.graphstate(graph)))
    dest = zeros(Float32, 2)

    err = try
        InteractiveIsing.calculate!(dest, d_sH(), Clamping(), graph; backend)
        nothing
    catch e
        e
    end

    @test err isa ArgumentError
    @test occursin("does not implement", sprint(showerror, err))
end
