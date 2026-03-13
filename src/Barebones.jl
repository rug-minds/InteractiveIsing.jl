export BareGraph, simulateBare, stop, start

using GLMakie
mutable struct BareGraph{T}
    const state::Vector{T}
    const layers::Vector{Tuple{Int32,Int32}}
    const adj::SparseMatrixCSC{T,Int32}
    const bias::Vector{T}
    temp::T
    shouldrun::Bool
    updates::UInt
end
stop(g::BareGraph) = g.shouldrun = false
Processes.start(g::BareGraph) = g.shouldrun = true


function BareGraph(T, layers, adj, bias)
    length = sum(map(x -> x[1]*x[2], layers))
    state = rand(length)*2 .- 1
    layers = convert.(Tuple{Int32,Int32}, layers)
    return BareGraph{T}(state, layers, adj, bias, T(1), true, 0)
end

let last_two = CircularBuffer{UInt}(2), times = CircularBuffer{UInt}(30), update_deltas = AverageCircular(Int, 30)
    push!(last_two, 0)
    push!(last_two, 0)
    global function ups!(obs, g)
        push!(last_two, g.updates)
        push!(times, time_ns())

        delta = last_two[2] == 0 ? 0 : last_two[2] - last_two[1]
        push!(update_deltas, delta)
        a = avg(update_deltas)
        obs[] = a / (times[end] - times[1]) * 1e9  # Convert to seconds
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

    updates_per_second = Observable(0f0)
    upf_label = lift(x -> "Updates per second: $x", updates_per_second)


    w = new_window()
    f = w.f
    grid = GridLayout(f[1,1])
    ax = Axis(grid[1, 1], aspect = 1)
    Label(grid[2,1], upf_label, tellwidth = false)

    image!(ax, img_ob, colormap = :thermal, fxaa = false, interpolate = false)
    # d = display(f)
    
    #Label should take 1/5 of screenspace
    # rowsize!(grid, 2, Relative(0.2))


    rng = MersenneTwister(1234)
    iterator = UnitRange{Int32}(1:length(g.state))
    task = Threads.@spawn loop(g, rng, iterator)
    updatefunc = () -> begin notify(img_ob); ups!(updates_per_second, g) end
    timer = Timer(t -> updatefunc(), 0, interval = 1/30)
    return timer, task
end

function loop(g::G, rng, iterator) where G
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