abstract type ProcessAlgorithm end
struct TrianglePulseProc <: ProcessAlgorithm end

function TrianglePulseProc(args)
    (;proc, triggers, x, y) = args

        # Waits so that the pulse is applied every tstep time
    while time() - t_i < tstep
        
    end
    # Sets the bfield
    setparam!(g[1], :b, pulse[triggeridx(triggers)])

    # setparam!(g[1][2], :b, pulse[idx], true)
    # setparam!(g[1][max_z], :b, pulse[idx], true)
    # setparam!(g[1][max_z-1], :b, pulse[idx], true)

    # note down the time
    t_i = time()
    
    push!(y, sum(state(g)))
        
    # Send the pulse amplitude to the observable
    push!(x, pulse[triggeridx(triggers)])
end

function InteractiveIsing.prepare(::TrianglePulseProc, args)
    (;amp, triggers, npulse) = args
    
    steps = maxsteps(triggers)

    first = LinRange(0, amp, floor(Int,steps/(4*npulse)))
    second = LinRange(amp, 0, floor(Int,steps/(4*npulse)))
    third = LinRange(0, -amp, floor(Int,steps/(4*npulse)))
    fourth = LinRange(-amp, 0, floor(Int,steps/(4*npulse)))
    pulse = vcat(first, second, third, fourth)

    pulse = repeat(pulse, npulse)

    return pulse
end

IsingAndMag = CompositeAlgorithm( (Metropolis, TrianglePulseProc), (1,100) )
