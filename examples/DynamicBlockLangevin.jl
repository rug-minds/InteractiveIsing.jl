using InteractiveIsing
using InteractiveIsing.Processes
using Statistics

# Run with:
#     julia --project examples/DynamicBlockLangevin.jl
#
# DynamicBlockLangevin chooses a fresh derivative block size for every cached
# block cycle:
#     m ~ Uniform(1:min(MaxBlockSize, n_active))
#
# Each step then attempts one spin update from that cached block.

const SIDE = 32
const NSTEPS = 2_000
const TEMPERATURE = 1.5f0
const MAX_BLOCKSIZE = 32
const STEPSIZE = 0.02f0

const NN_WEIGHTS = @WG (; dr) -> dr == 1 ? 1f0 : 0f0 NN = 1

function make_graph()
    g = IsingGraph(
        SIDE,
        SIDE,
        Continuous(),
        NN_WEIGHTS,
        StateSet(-1f0, 1f0),
        Ising(c = ConstVal(1f0), b = 0f0),
        periodic = (:x, :y),
    )
    temp!(g, TEMPERATURE)
    return g
end

function run_steps!(g, algorithm; nsteps = NSTEPS)
    context = Processes.init(algorithm, (; model = g))
    accepted = 0
    attempted = 0
    block_sizes = Int[]

    for _ in 1:nsteps
        out = Processes.step!(algorithm, context)
        context = merge(context, out)

        accepted += out.accepted
        attempted += out.attempted
        push!(block_sizes, out.block_size)
    end

    return (
        acceptance_rate = accepted / attempted,
        mean_block_size = mean(block_sizes),
        block_size_range = extrema(block_sizes),
        final_state_range = extrema(state(g)),
    )
end

fixed_graph = make_graph()
dynamic_graph = make_graph()
copyto!(state(dynamic_graph), state(fixed_graph))

fixed = BlockLangevin(
    stepsize = STEPSIZE,
    block_size = MAX_BLOCKSIZE,
    adjusted = true,
)

dynamic = DynamicBlockLangevin(
    stepsize = STEPSIZE,
    max_blocksize = MAX_BLOCKSIZE,
    adjusted = true,
)

println("Fixed block Langevin:")
display(run_steps!(fixed_graph, fixed))

println("\nDynamic block Langevin:")
display(run_steps!(dynamic_graph, dynamic))
