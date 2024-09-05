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
        M[] = [sum(state(g))]
    end
    if !isnothing(PulseAmp)
        PulseAmp[] = [0.]
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
    for idx in 1:steps
        while time() - t_i < tstep
            
        end
        setparam!(g[1], :b, pulse[idx], true)
        # setparam!(g[1][2], :b, pulse[idx], true)
        # setparam!(g[1][max_z], :b, pulse[idx], true)
        # setparam!(g[1][max_z-1], :b, pulse[idx], true)

        t_i = time()
        
        if !isnothing(M)
            push!(M[], sum(state(g)))
        end

        if !isnothing(PulseAmp)
            push!(PulseAmp[], pulse[idx])
            notify(PulseAmp)
        end

        if !isnothing(ax)
            autolimits!(ax)
        end
        println("Pulse step: ", idx, " of ", steps)
    end

    println("Done")

end

Marray = Observable([0.])
Pulsearray = Observable([0.])
# B VS M
ax = new_axis()
lines!(ax, Pulsearray, Marray)
@async TrianglePulseB(g, 20, 2, 2000, npulse = 2, M = Marray, PulseAmp = Pulsearray, ax = ax)
