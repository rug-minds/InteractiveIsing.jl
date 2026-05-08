using InteractiveIsing
using InteractiveIsing.Processes

# Nearest-neighbour ferromagnetic coupling.
function isingweights(; dr::R) where {R}
    return dr == 1 ? 1f0 : 0f0
end

wg = @WG isingweights NN = 1

# Unbounded spins: StateSet(-Inf32, Inf32) removes hard boundary rejections.
# A uniform random initial state is impossible on an infinite interval, so the
# constructor initializes unbounded continuous layers from local potentials when
# possible. For H_local(x) = a*x^2 + l*x it starts at -l/(2a). If no positive
# local quadratic well is present, it starts at zero.
#
# The Hamiltonian below uses a positive local quadratic well plus a magnetic
# field. With c=16, localpotential=1, and b=4, the isolated local minimum is
# x=b/(2c)=0.125.

g = IsingGraph(
    30, 30, 10,
    Continuous(),
    wg,
    LatticeConstants(1f0, 1f0, 1f0),
    StateSet(-Inf32, Inf32),          # unbounded spins
    Ising(c = ConstVal(16f0), localpotential = ConstFill(1f0), b = ConstFill(4f0)),
    periodic = (:x, :y),
)

temp!(g, 18.64f0)
state(g) .= 0f0

# GlobalLangevin refreshes all active-spin derivatives, then attempts one spin
# update per step. With unbounded spins there is no out-of-bounds rejection, so
# the acceptance rate is controlled only by the MH ratio.
algorithm = GlobalLangevin(
    stepsize  = 0.001f0,
    adjusted  = true,
)

function langevin_status(g)
    ps = processes(g)
    p = process(g)
    ctx = isnothing(p) ? nothing : p.context[:GlobalLangevin_1]

    return (
        temp = temp(g),
        stepsize = isnothing(ctx) ? nothing : ctx.stepsize[],
        adjusted = isnothing(ctx) ? nothing : ctx.adjusted[],
        group_steps = isnothing(ctx) ? nothing : ctx.group_steps[],
        process_count = length(ps),
        ticks = isnothing(p) ? 0 : Processes.getticks(p),
        extrema = extrema(state(g)),
        maxabs = maximum(abs, state(g)),
        finite = all(isfinite, state(g)),
    )
end

interface(g)
p = createProcess(g, algorithm)
