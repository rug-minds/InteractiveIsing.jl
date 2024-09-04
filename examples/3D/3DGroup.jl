using InteractiveIsing, LoopVectorization, SparseArrays

g = IsingGraph(10,10,10)

simulate(g)
function weightfunc(dx,dy,dz)
    prefac = 1
    dr2 = (2*dx)^2+(2*dy)^2+dz^2
    if dy != 0 || dx != 0
        prefac *= -1
    end
    return prefac/dr2
end


wg = @WG "(dx,dy,dz) -> weightfunc(dx,dy,dz)" NN = (1,1,1)

genAdj!(g[1], wg)

setparam!(g[1], :b, 2, true)

#TODO Optimize this
function scaleWeights(g::IsingGraph{T}, idx, scale::T) where T
    gadj = adj(g)
    @turbo for ptr in nzrange(gadj, idx)
        gadj.nzval[ptr] *= scale
    end
end


function TrianglePulseB(g, t, amp = 1, steps = 1000)
    first = LinRange(0, amp, floor(Int,steps/4))
    second = LinRange(amp, 0, floor(Int,steps/4))
    third = LinRange(0, -amp, floor(Int,steps/4))
    fourth = LinRange(-amp, 0, floor(Int,steps/4))
    pulse = vcat(first, second, third, fourth)

    tstep = t/steps

    t_i = time()
    for idx in 1:steps
        while time() - t_i < tstep
            
        end
        setparam!(g, :b, pulse[idx], true)
        t_i = time()
    end

end

Threads.@spawn TrianglePulseB(g, 10, 5)