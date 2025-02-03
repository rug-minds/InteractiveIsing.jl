using InteractiveIsing, Processes
g = IsingGraph(500,500, type = Discrete)
wg = @WG "dr -> dr == 1 ? 1 : 0" NN=1
genAdj!(g[1], wg)

struct CalcSusceptibility <: ProcessAlgorithm end

function Processes.prepare(::CalcSusceptibility, args)
    M2s = []
    processsizehint!(args, M2s)
    (;M2s)
end

function CalcSusceptibility(args)
    (;M, M2s) = args
    push!(M2s, M[]^2)
    return
end

function Processes.cleanup(::CalcSusceptibility, args)
    (;M2s) = args
    avg = sum(M2s)/length(M2s)
    σ2 = sum(x -> (x - avg)^2, M2s)/length(M2s)
    return (;avg, σ2)
end

temp(g, 2.27)
interface(g)
SweepSusceptibility = CompositeAlgorithm( (CheckeredSweepMetropolis, CalcSusceptibility), (1, 500^2))
createProcess(g, lifetime = 500^2*1000)
createProcess(g, algorithm = SweepSusceptibility, lifetime = 500^2 * 100)
