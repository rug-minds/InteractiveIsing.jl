using InteractiveIsing

g = IsingGraph(40,40,40)

simulate(g)

wg = @WG "(dx,dy,dz) -> 1/(dx^2+dy^2+dz^2)" NN = (2,2,5)
genAdj!(g[1], wg)

setParam!(g[1], :b, 2, true)


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
        setParam!(g, :b, pulse[idx], true)
        t_i = time()
    end

end

Threads.@spawn TrianglePulseB(g, 10, 5)