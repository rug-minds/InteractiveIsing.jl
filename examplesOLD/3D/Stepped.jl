using InteractiveIsing, LoopVectorization, Processes
Processes.debug_mode(false)

N = 40
g = IsingGraph(N, N, N, type = Discrete)

# createProcess(g)
# interface(g)
simulate(g)

function weightfunc(dx,dy,dz)
    prefac = 1
    dr2 = (dx)^2+(dy)^2+dz^2
    if abs(dy) > 0 || abs(dx) > 0
        prefac *= 1
    end
    if dr2 != 1
        return 0
    end
    return prefac
end


wg = @WG "(dx,dy,dz) -> weightfunc(dx,dy,dz)" NN = (1,1,3)

genAdj!(g[1], wg)

setparam!(g[1], :b, 0, true)


struct TrianglePulseB end

function Processes.prepare(::TrianglePulseB, args)
    (;amp, numpulses) = args

    steps = num_calls(args)

    first = LinRange(0, amp, floor(Int,steps/(4*numpulses)))
    second = LinRange(amp, 0, floor(Int,steps/(4*numpulses)))
    third = LinRange(0, -amp, floor(Int,steps/(4*numpulses)))
    fourth = LinRange(-amp, 0, floor(Int,steps/(4*numpulses)))
    pulse = vcat(first, second, third, fourth)

    pulse = repeat(pulse, numpulses)

    x = Float32[]
    y = Float32[]
    all_Es = Float32[]

    processsizehint!(args, x)
    processsizehint!(args, y)
    processsizehint!(args, all_Es)
    

    return (;pulse, x, y, all_Es)
end

function (::TrianglePulseB)(args)
    (;pulse, M, x, y, B, all_Es, total_E) = args
    pulse_val = pulse[algo_loopidx(args)]
    # setparam!(g[1], :b, pulse_val)

    B[] = pulse_val

    push!(x, pulse_val)
    push!(y, M[])
    push!(all_Es, total_E[])
    
end

fullsweep = N^3
compalgo = CompositeAlgorithm((MetropolisGB, TrianglePulseB), (1, 50))

createProcess(g, compalgo, lifetime = 10000*fullsweep, amp = 3, numpulses = 2)

# createProcess(g, compalgo, amp = 1)




