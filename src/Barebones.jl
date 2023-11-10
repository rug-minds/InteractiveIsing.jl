export BareGraph, simulateBare, stop, start
mutable struct BareGraph{T}
    const state::Vector{T}
    const layers::Vector{Tuple{Int32,Int32}}
    const adj::SparseMatrixCSC{T,Int32}
    const bias::Vector{T}
    temp::T
    shouldrun::Bool
    updates::Int
end
stop(g::BareGraph) = g.shouldrun = false
start(g::BareGraph) = g.shouldrun = true


function BareGraph(T, layers, adj, bias)
    length = sum(map(x -> x[1]*x[2], layers))
    state = rand(length)*2 .- 1
    layers = convert.(Tuple{Int32,Int32}, layers)
    return BareGraph{T}(state, layers, adj, bias, T(1), true, 0)
end
let frames = 1
    global function clearupdates(obs, circavg, g)
        push!(circavg, g.updates)
        obs.val = Float32(avg(circavg))
        if frames == 60
            notify(obs)
            frames = 1
        else
            frames += 1
        end
        g.updates = 0
    end
end

function simulateBare(g::BareGraph, layeridx)
    previous_layer_lengths = sum(map(x -> x[1]*x[2], g.layers[1:layeridx-1]))
    layerlength = g.layers[layeridx][1]*g.layers[layeridx][2]
    layeridxs = previous_layer_lengths+1:previous_layer_lengths+layerlength
    println("Previous layer lengths: $previous_layer_lengths")
    println("Layer length: $layerlength")
    println("Layer idxs: $layeridxs")
    stateview = reshape((@view g.state[layeridxs]), g.layers[layeridx][1], g.layers[layeridx][2])
    img_ob = Observable(stateview)

    updates_per_frame = Observable(0f0)
    upf_label = lift(x -> "Updates per frame: $x", updates_per_frame)

    circavg = AverageCircular(Float32, 60)

    f = Figure();
    grid = GridLayout(f[1,1])
    ax = Axis(f[1, 1], aspect = 1)
    image!(ax, img_ob, colormap = :thermal, fxaa = false, interpolate = false)
    d = display(f)
    Label(grid[2,1], upf_label, tellwidth = false)
    #Label should take 1/5 of screenspace
    # rowsize!(grid, 2, Relative(0.2))


    rng = MersenneTwister(1234)
    iterator = UnitRange{Int32}(1:length(g.state))
    task = Threads.@spawn loop(g, rng, iterator)
    updatefunc = () -> begin notify(img_ob); clearupdates(updates_per_frame, circavg, g) end
    timer = Timer(t -> updatefunc(), 0, interval = 1/60)
    return timer, task
end

function loop(@specialize(g), rng, iterator)
    while g.shouldrun
        idx = rand(rng, iterator)
        BareMonteCarlo(g, rng, idx)
        g.updates += 1
        GC.safepoint()
    end
end

@inline function BareMonteCarlo(@specialize(g::BareGraph{T}), rng, idx) where T
    newstate = rand(rng)*T(2) - one(T)
    β = one(T)/g.temp
    ΔE = (newstate-g.state[idx])BaredE(g, idx)
    if (ΔE < zero(T) || rand(rng, Float32) < exp(-β*ΔE))
        @inbounds g.state[idx] = newstate 
    end
end

@inline function BaredE(@specialize(g::BareGraph{T}), idx) where T
    efactor = zero(T)
    @turbo for ptr in nzrange(g.adj, idx)
        efactor -= g.adj.nzval[ptr]*g.state[g.adj.rowval[ptr]]
    end
    return efactor -= g.bias[idx]
end