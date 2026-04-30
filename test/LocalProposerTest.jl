using Test
using InteractiveIsing
using Random

@testset "LocalProposer" begin
    continuous_layer = Layer(8, Continuous(), StateSet(-1.0f0, 1.0f0))
    continuous_graph = IsingGraph(InteractiveIsing.EmptyHamiltonian(), continuous_layer)
    continuous_proposer = InteractiveIsing.LocalProposer(continuous_graph, 0.05f0)

    continuous_proposals = [rand(MersenneTwister(i), continuous_proposer) for i in 1:20]
    @test all(p -> p isa FlipProposal, continuous_proposals)
    @test all(p -> -1.0f0 <= p.to_val <= 1.0f0, continuous_proposals)

    state(continuous_graph) .= 0.99f0
    boundary_proposals = [rand(MersenneTwister(i), continuous_proposer) for i in 21:80]
    @test all(p -> -1.0f0 <= p.to_val <= 1.0f0, boundary_proposals)
    @test any(p -> p.to_val != p.from_val, boundary_proposals)

    discrete_layer = Layer(8, Discrete(), StateSet(-1.0f0, 0.0f0, 1.0f0))
    discrete_graph = IsingGraph(InteractiveIsing.EmptyHamiltonian(), discrete_layer)
    discrete_proposer = InteractiveIsing.LocalProposer(discrete_graph, 1)

    discrete_proposals = [rand(MersenneTwister(i), discrete_proposer) for i in 1:20]
    @test all(p -> p isa FlipProposal, discrete_proposals)
    @test all(p -> p.to_val in stateset(discrete_graph[1]), discrete_proposals)

    parsed_graph = IsingGraph(8, LocalProposer(1), Discrete(), StateSet(-1.0f0, 1.0f0))
    parsed_algo = InteractiveIsing.Processes.getalgo(parsed_graph.default_algorithm)
    parsed_context = InteractiveIsing.Processes.init(parsed_algo, (;model = parsed_graph))
    parsed_output = InteractiveIsing.Processes.step!(parsed_algo, parsed_context)

    @test parsed_output.proposal isa FlipProposal
    @test parsed_output.proposal.to_val in stateset(parsed_graph[1])
end
