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

function TrianglePulseB(g, t, amp = 1, steps = 1000; npulse = 1)
 
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

    process = linesprocess(length(pulse)) do args
        # println("Working on thread ", Threads.threadid())
        (;proc, x, y) = args

        # Waits so that the pulse is applied every tstep time
        while time() - t_i < tstep
            
        end

        # Sets the bfield
        setparam!(g[1], :b, pulse[loopidx(proc)], true)


        # setparam!(g[1][2], :b, pulse[idx], true)
        # setparam!(g[1][max_z], :b, pulse[idx], true)
        # setparam!(g[1][max_z-1], :b, pulse[idx], true)

        # note down the time
        t_i = time()
        
        push!(y, sum(state(g)))

        # Send the pulse amplitude to the observable
        push!(x, pulse[loopidx(proc)])

    end

    return process


end

w = lines_window(TrianglePulseB(g, 0.01, 50, 2000, npulse = 2))
