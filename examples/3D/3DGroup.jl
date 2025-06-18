using InteractiveIsing, LoopVectorization, Processes

g = IsingGraph(40,40,40, type = Discrete)
 
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

# setparam!(g[1], :b, 0, true)

#TODO Optimize this
function scaleWeights(g::IsingGraph{T}, idx, scale::T) where T
    gadj = adj(g)
    @turbo for ptr in nzrange(gadj, idx)
        gadj.nzval[ptr] *= scale
    end
end

using GLMakie

struct TrianglePulseB end

function Processes.prepare(::TrianglePulseB, args)
    (;lifetime, amp, numpulses) = args
    max_z = size(g[1], 3)

    steps = Processes.repeats(lifetime)
    first = LinRange(0, amp, floor(Int,steps/(4*numpulses)))
    second = LinRange(amp, 0, floor(Int,steps/(4*numpulses)))
    third = LinRange(0, -amp, floor(Int,steps/(4*numpulses)))
    fourth = LinRange(-amp, 0, floor(Int,steps/(4*numpulses)))
    pulse = vcat(first, second, third, fourth)

    pulse = repeat(pulse, numpulses)

    x = Float64[]
    y = Float64[]
    processsizehint!(args, x)
    processsizehint!(args, y)
   return (;pulse, x, y, tstep = args.tstep) |> add_timetracker
end

function TrianglePulseB(args)
    (;proc, pulse, tstep, x, y) = args
    wait(args, tstep)


    setparam!(g[1], :b, pulse[loopidx(proc)])

    push!(y, sum(state(g)))            
    push!(x, pulse[loopidx(proc)])
end

# InteractiveIsing.change_args!(w, tstep = 0.001, amp = 2)
