using InteractiveIsing, Processes
# using InteractiveIsing.Processes

struct CalcSusceptibility <: ProcessAlgorithm end

function Processes.prepare(::CalcSusceptibility, args)
    M2s = []
    processsizehint!(args, M2s)
    (;M2s)
end

function CalcSusceptibility(args)
    (;g, gstate, M, M2s) = args
    push!(M2s, M[]^2)
    return
end

function Processes.cleanup(::CalcSusceptibility, args)
    (;M2s) = args
    avg = sum(M2s)/length(M2s)
    σ2 = sum(x -> (x - avg)^2, M2s)/length(M2s)
    return (;avg, σ2)
end

N = 500
g = IsingGraph(N,N, type = Discrete)
wg = @WG "dr -> dr == 1 ? 1 : 0" NN=1
genAdj!(g[1], wg)

temp(g, 2.27)
# interface(g)

const sweepsteps = N^2

Equilibration = SubRoutine(Metropolis, 1e6*sweepsteps)
SweepSusceptibility = CompositeAlgorithm( (CheckeredSweepMetropolis, CalcSusceptibility), (1, sweepsteps))
SweepRoutine = SubRoutine(SweepSusceptibility, 1e6*sweepsteps)
routine = Routine(Equilibration, SweepRoutine)

createProcess(g, routine)