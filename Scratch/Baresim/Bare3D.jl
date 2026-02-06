using LoopVectorization, SparseArrays, GLMakie

abstract type StateType end
struct Continuous <: StateType end
struct Discrete <: StateType end

mutable struct SimpleGraph{StateType, Dim, GT, ADT}
    shouldrun::Bool
    const state::Vector{GT}
    const adj::ADT
    temp::GT
    size::NTuple{Dim, Int}
    steps::Int
    mw::Any
end
statetype(sg::SimpleGraph{ST}) where ST = ST
Base.eltype(sg::SimpleGraph) = eltype(sg.state)

samplefullstate(length, ::Type{Continuous}, type = Float32) = type(2).*rand(type, length) .- type(1)
samplefullstate(length, ::Type{Discrete}, type = Float32) = rand([type(-1), type(1)], length)

samplestate(::Type{Continuous}, eltype) = eltype(2).*rand(eltype) .- eltype(1)
samplestate(::Type{Discrete}, eltype) = rand([eltype(-1), eltype(1)])

SimpleGraph(size::NTuple{N, <:Integer}; beta = 1, adj = spzeros(Int32, size...), statetype = Continuous, type = Float32) where N = SimpleGraph{statetype, N, type, typeof(adj)}(true, samplefullstate(prod(size), statetype, type), adj, beta, size, 1, nothing) 

function montecarlo_step(@specialize(g::SimpleGraph{StateType, Dim, Eltype}), adj) where {StateType, Dim, Eltype}
    beta = 1/g.temp

    #choose idx
    idx = rand(1:length(g.state))

    oldstate = g.state[idx]
    newstate = samplestate(StateType, Eltype)

    #compute energy
    dE = (oldstate-newstate) * compute_energy(g.state, adj, idx, Eltype)

    #flip
    if dE < 0 || rand() < exp(-beta*dE)
        g.state[idx] = newstate
    end

end

function compute_energy(state, adj, idx, Eltype)
    energy = Eltype(0)
    @turbo for ptr in nzrange(adj, idx)
        j = adj.rowval[ptr]
        outstate = state[j]
        weight = adj.nzval[ptr]
        energy += outstate * weight
    end
    return energy
end

function mainloop(@specialize(g)) 
    while g.shouldrun
        @inline montecarlo_step(g, g.adj)
        g.steps += 1
    end
end

function stop(g)
    g.shouldrun = false
end

struct SimpleMakieWindow
    fig
    ax
    lines
    timer
    screen
end

idx2ycoord(size::NTuple{3,T}, idx) where {T} = (T(idx)-T(1)) % size[1] + T(1)
idx2xcoord(size::NTuple{3,T}, idx) where {T} = (floor(T, (idx-T(1))/size[1])) % size[2] + T(1)
idx2zcoord(size::NTuple{3,T}, idx) where {T} = floor(T, (idx-T(1))/(size[1]*size[2])) + T(1)

include("../../src/Utils/CastVec.jl")
include("../../src/Utils/AverageCircular.jl")
function SimpleMakieWindow(g::SimpleGraph{ST, Dim}) where {ST, Dim}
    fig = Figure()
    obs = nothing
    if Dim == 2
        ax = Axis(fig[1, 1])
        # Image with reshaped observable
        reshapestate = reshape(g.state, g.size...)
        obs = Observable(reshapestate)
        image!(ax, obs, colormap = :thermal, fxaa = false, interpolate = false)
    elseif Dim == 3
        # #
        # unsafe_view = create_unsafe_vector(@view state(g)[graphidxs(layer)])
        #     mp["obs"] = Observable(CastVec(Float64, unsafe_view))
        #     # mp["obs"] = Observable(@view state(g)[graphidxs(layer)])
        # else
        #     mp["obs"] = color
        # end
       
        # sz = size(layer)
        # allidxs = [1:length(state(layer));]
        # xs = idx2xcoord.(Ref(sz), allidxs)
        # ys = idx2ycoord.(Ref(sz), allidxs)
        # zs = idx2zcoord.(Ref(sz), allidxs)
        
        # mp["image"] = meshscatter!(ax, xs, ys, zs, markersize = 0.2, color = mp["obs"], colormap = colormap)
        # unsafe_view = create_unsafe_vector(state(g))
        obs = Observable(CastVec(Float64, g.state))
        
        ax = Axis3(fig[1, 1])
        xs = idx2xcoord.(Ref(g.size), 1:length(g.state))
        ys = idx2ycoord.(Ref(g.size), 1:length(g.state))
        zs = idx2zcoord.(Ref(g.size), 1:length(g.state))
        meshscatter!(ax, xs, ys, zs, markersize = 0.2, color = obs, colormap = :thermal)
    end

    avgc = AverageCircular(Int, 60)
    upf = Observable(0.)
    # Make Label underneath
    grid = GridLayout(fig[2, 1])
    text = lift(x -> "Updates per frame: $(round(x, digits = 2))", upf)
    grid[1,1] = Label(fig, text, halign = :right, tellwidth = false)

    last_steps = 0
    frame = 1
    function updates_per_frame()
        this_steps = g.steps
        push!(avgc, this_steps - last_steps)
        last_steps = this_steps
        if frame >= 60
            frame = 1
            upf[] = avg(avgc)
        else
            frame += 1
        end
    end

    #Notify
    timer = Timer((timer)-> (notify(obs); updates_per_frame()), 0, interval = 1/60)

    screen = GLMakie.Screen()
    d = display(screen, fig)
    SimpleMakieWindow(fig, ax, lines, timer, GLMakie.Screen())
end

Base.close(w::SimpleMakieWindow) = close(w.timer)
function Base.close(bg::SimpleGraph)
    bg.shouldrun = false
    close(bg.mw)
    GLMakie.GLFW.SetWindowShouldClose(GLMakie.to_native(bg.mw.screen), true)
end

bg = SimpleGraph(size(g[1]), adj = adj(g))

function start_baresim(bg)
    bg.shouldrun = true
    Threads.@spawn mainloop(bg)
    bg.mw = SimpleMakieWindow(bg)
end

mw = start_baresim(bg)
