using InteractiveIsing, LoopVectorization, SparseArrays

g = IsingGraph(40,40,40)

simulate(g)

function weightfunc(dx,dy,dz)
    prefac = 1
    dr2 = (2*dx)^2+(2*dy)^2+dz^2
    if abs(dy) > 0 || abs(dx) > 0
        prefac *= -1
    end
    return prefac/dr2
end


wg = @WG "(dx,dy,dz) -> weightfunc(dx,dy,dz)" NN = (1,1,2)

genAdj!(g[1], wg)

setparam!(g[1], :b, 0, true)

#TODO Optimize this
function scaleWeights(g::IsingGraph{T}, idx, scale::T) where T
    gadj = adj(g)
    @turbo for ptr in nzrange(gadj, idx)
        gadj.nzval[ptr] *= scale
    end
end

using GLMakie

function TrianglePulseB(g, t, amp = 1, steps = 1000; npulse = 1, M = nothing, PulseAmp = nothing, ax = nothing)
    if !isnothing(M)
        deleteat!(M, 2:length(M))
        M[1] = sum(state(g))
    end
    if !isnothing(PulseAmp)
        deleteat!(PulseAmp, 2:length(PulseAmp))
        PulseAmp[1] = 0
    end

    max_z = size(g[1], 3)

    first = LinRange(0, amp, floor(Int,steps/(4*npulse)))
    second = LinRange(amp, 0, floor(Int,steps/(4*npulse)))
    third = LinRange(0, -amp, floor(Int,steps/(4*npulse)))
    fourth = LinRange(-amp, 0, floor(Int,steps/(4*npulse)))
    pulse = vcat(first, second, third, fourth)

    pulse = repeat(pulse, npulse)


    tstep = t/steps

    t_i = time()

    # println("Pulse length: ", length(pulse))

    process = makeprocess(length(pulse)) do args
        # println("Working on thread ", Threads.threadid())
        (;proc) = args

        # Waits so that the pulse is applied every tstep time
        while time() - t_i < tstep
            
        end

        # Sets the bfield
        setparam!(g[1], :b, pulse[loopidx(proc)], true)

        # println("Pulse step: ", loopidx(proc), " of ", steps)

        # setparam!(g[1][2], :b, pulse[idx], true)
        # setparam!(g[1][max_z], :b, pulse[idx], true)
        # setparam!(g[1][max_z-1], :b, pulse[idx], true)

        # note down the time
        t_i = time()
        
        # Send the total magnetization to the observable
        # if !isnothing(M)
            push!(M, sum(state(g)))
            println("Magnetization: ", sum(state(g)))
            println("Length of M: ", length(M))
        # end

        # Send the pulse amplitude to the observable
        # if !isnothing(PulseAmp)
            push!(PulseAmp, pulse[loopidx(proc)])
            println("Pulse amplitude: ", pulse[loopidx(proc)])
            println("Length of PulseAmp: ", length(PulseAmp))
        # end

        # Update plot axis
        if !isnothing(ax)
            autolimits!(ax)
        end
        # println("Pulse step: ", loopidx(p), " of ", steps)
    end

    return process


end

const y = [0.]
const x = [0.]
w = lines_window(x, y, process = TrianglePulseB(g, 2, 2, 50, npulse = 2, M = y, PulseAmp = x))
