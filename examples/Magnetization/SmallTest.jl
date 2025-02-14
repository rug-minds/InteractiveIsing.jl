using InteractiveIsing, Processes
Processes.debug_mode(true)
# using InteractiveIsing.Processes
# Processes.est_remaining(g::IsingGraph) = est_remaining(process(g))


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

    return (;args..., avg, σ2)
end

# Ns = [500]
# gs = IsingGraph[]

wg = @WG "dr -> dr == 1 ? 1 : 0" NN=1

N = 500
# for N in Ns
    g = IsingGraph(N,N, type = Discrete, weights = wg)
    # push!(gs, g)

    temp(g, 2.27)

    sweepsteps = N^2
    eqsteps = 5000
    datasteps = 1000000

    # Equilibration = SubRoutine(Metropolis, 1e5*sweepsteps)
    Equilibration = SubRoutine(Metropolis, eqsteps*sweepsteps)
    SweepSusceptibility = CompositeAlgorithm( (CheckeredSweepMetropolis, CalcSusceptibility), (1, 500^2))
    SweepRoutine = SubRoutine(SweepSusceptibility, Int(datasteps*sweepsteps))
    SweepRoutine2 = SubRoutine(SweepSusceptibility, 1e6*sweepsteps)

    routine2 = Routine(Equilibration, SweepRoutine2)
    p2 = Process(routine2; g)
    preparedata!(p2);

    routine = Routine(Equilibration, SweepRoutine)
    p = Process(routine; g)
    preparedata!(p);

    # createProcess(g, routine)
# end



