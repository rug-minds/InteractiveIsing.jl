


using InteractiveIsing, Processes

struct CalcSusceptibility <: ProcessAlgorithm end

function incs_per_sec(p::Process)
    loopidx(p) / runtime(p)
end

# Runs one time at the beginning
function Processes.init(::CalcSusceptibility, args)
    M2s = Int[]
    processsizehint!(args, M2s)
    return (;M2s)
end

# Once per simulation step
function (::CalcSusceptibility)(args)
    (;g, gstate, M, M2s) = args

    push!(M2s, M[]^2)
    return
end

# Runs one time at the end
function Processes.cleanup(::CalcSusceptibility, args)
    (;M2s) = args
    avg = sum(M2s)/length(M2s)
    σ2 = sum(x -> (x - avg)^2, M2s)/length(M2s)

    

    return (;avg, σ2)
end

N = 500
wg = @WG "dr -> dr == 1 ? 1 : 0" NN=1
g = IsingGraph(N,N, type = Discrete, weights = wg)
interface(g)



eqsteps = 1e2
temp(g, 2.27)
sweeps = 1e2

onesweep = N^2

Equilibration = CheckeredSweepMetropolis
SweepSusceptibility = CompositeAlgorithm( (CheckeredSweepMetropolis, CalcSusceptibility), (1, onesweep))

routine = Routine((Equilibration, SweepSusceptibility), (eqsteps*onesweep, sweeps*onesweep))
using Random
createProcess(g, routine, overrides = (;rng = MersenneTwister()))

final_context = getcontext(process(g))
