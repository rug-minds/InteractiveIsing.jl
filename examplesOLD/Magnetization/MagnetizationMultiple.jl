using InteractiveIsing, Processes
# using InteractiveIsing.Processes

struct CalcSusceptibility <: ProcessAlgorithm end

function Processes.init(::CalcSusceptibility, args)
    M2s = Int[]
    processsizehint!(args, M2s)
    (;M2s)
end

function (::CalcSusceptibility)(args)
    (;g, M, M2s) = args
    push!(M2s, M[]^2)
    return
end

function Processes.cleanup(::CalcSusceptibility, args)
    (;M2s) = args
    avg = sum(M2s)/length(M2s)
    σ2 = sum(x -> (x - avg)^2, M2s)/length(M2s)
    return (;avg, σ2)
end

graphs = IsingGraph[]

Ns = [20,50,100,200,500]
for N in Ns
    g = IsingGraph(N,N, type = Discrete)
    wg = @WG "dr -> dr == 1 ? 1 : 0" NN=1
    genAdj!(g[1], wg)
    push!(graphs, g)

    eqsteps = 1e4
    temp(g, 2.27)
    sweeps = 1e5

    sweepsteps = N^2

    Equilibration = CheckeredSweepMetropolis
    SweepSusceptibility = CompositeAlgorithm( (CheckeredSweepMetropolis, CalcSusceptibility), (1, sweepsteps))

    routine = Routine((Equilibration, SweepSusceptibility), floor.(Int,(eqsteps*sweepsteps, sweeps*sweepsteps)))
    
    createProcess(g, routine)
end


function incs_per_sec(p::Process)
    loopidx(p) / runtime(p)
end

