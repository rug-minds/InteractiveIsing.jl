using InteractiveIsing

g = simulate(28,28, type = Continuous, set = (0f0, 1f0), precision = Float64)
addLayer!(g, 10,50, type = Discrete, set = (0f0, 1f0))
addLayer!(g, 2,5, type = Discrete, set = (0f0, 1f0))

using DelimitedFiles, CSV, DataFrames, SparseArrays
v_bias = (DataFrame(CSV.File("examples/bin/train_b_x.csv"))[:,1])
h_bias = (DataFrame(CSV.File(("examples/bin/train_b_h.csv")))[:,1])
z_bias = (DataFrame(CSV.File(("examples/bin/train_b_z.csv")))[:,1])
w_vh = Matrix{Float64}(DataFrame(CSV.File(("examples/bin/train_W_xh.csv"))))
w_hz = transpose(Matrix{Float64}(DataFrame(CSV.File(("examples/bin/train_W_zh.csv")))))

function weights12_2_sparse(w1, w2)
    nstates = length(state(g))
    n_conns = length(w1)+length(w2)
    rowval = Int32[]
    colval = Int32[]
    nzval = Float64[]
    sizehint!(rowval, n_conns)
    sizehint!(colval, n_conns)
    sizehint!(nzval, n_conns)

    for i in 1:size(w1, 2)
        for j in 1:size(w1, 1)
                col = InteractiveIsing.idxLToG(j, g[1])
                row = InteractiveIsing.idxLToG(i, g[2])

                push!(rowval, row)
                push!(colval, col)
                push!(rowval, col)
                push!(colval, row)
                push!(nzval, w1[j,i])
                push!(nzval, w1[j,i])
        end
    end
    for i in 1:size(w2, 2)
        for j in 1:size(w2, 1)
                col = InteractiveIsing.idxLToG(j, g[2])
                row = InteractiveIsing.idxLToG(i, g[3])

                push!(rowval, row)
                push!(colval, col)

                push!(rowval, col)
                push!(colval, row)

                push!(nzval, w2[j,i])
                push!(nzval, w2[j,i]) 
        end
    end
    return sparse(rowval, colval, nzval, nstates, nstates)
end
sp_adj(g, weights12_2_sparse(w_vh, w_hz))
setB!(g[1], v_bias)
setB!(g[2], h_bias)
setB!(g[3], z_bias)

function updateRBM end
using LoopVectorization, StatsBase, Random

function InteractiveIsing.prepare(::typeof(updateRBM), g; kwargs...)
    def_kwargs = pairs((;g,
                        gstate = state(g),
                        gadj = sp_adj(g),
                        gparams = params(g),
                        iterator = stateiterator(g),
                        rng = MersenneTwister(),
                        gstype = stype(g),
                        its = aliveidxs.(layers(g))
                        # ΔEFunc = ΔEIsing,
                    ))
    return (;replacekwargs(def_kwargs, kwargs)...)
end

function sigmoid!(vec)
    for idx in eachindex(vec)
        vec[idx] = 1.0/(1.0+exp(-vec[idx]))
    end
end

function softmax!(vec)
    if isempty(vec)
        return
    end
    maxval = maximum(vec)
    for idx in eachindex(vec)
        vec[idx] = exp(vec[idx]-maxval)
    end
    sm = sum(vec)
    for idx in eachindex(vec)
        vec[idx] = vec[idx]/sm
    end
end
sigmoid(x) = 1.0./(1.0.+exp.(-x))
softmax(x) = exp.(x)./sum(exp.(x))

	# Sampling methods.
	#
function rbmstate!(vec)
    for idx in eachindex(vec)
        vec[idx] = 1.0*(rand() <= vec[idx])
    end
end

function choose!(vec)
    if isempty(vec)
        return
    end
    idx = sample(Weights(vec))
    vec .= 0
    vec[idx] = 1.0
end

function updateRBM(@specialize(args))
    (;g, gstate, gadj, gparams, iterator, rng, gstype, its) = args

    tempstate = sp_adj(g)*gstate
    v = @view tempstate[graphidxs(g[1])]
    h = @view tempstate[graphidxs(g[2])]
    z = @view tempstate[graphidxs(g[3])]

    sigmoid!(v)
    rbmstate!(v)

    sigmoid!(h)
    rbmstate!(h)

    softmax!(z)
    choose!(z)
    gstate .= tempstate
end

restart(g, algorithm = updateRBM)
createProcess(g, algorithm = updateRBM)


using GLMakie, DataStructures
# scene = Scene(resolution = (800, 800));
f = Figure();
ax = Axis(f[1, 1], aspect = 1)
d = display(GLMakie.Screen(), f)


const shitload = Matrix{AverageCircular{Float64}}(undef, 28, 28)
for idx in eachindex(shitload)
    shitload[idx] = AverageCircular(Float64, 1024)
end
avgs = zeros(Float32, 28, 28)
const img_ob = Observable(avgs)
image!(ax, img_ob, colormap = :thermal, fxaa = false, interpolate = false)

function updateshitload(shitload, state)
    for idx in eachindex(shitload)
        push!(shitload[idx], state[idx])
    end
end

function updateavgs(avgs, shitload)
    for idx in eachindex(avgs)
        avgs[idx] = avg(shitload[idx])
    end
end

const tShouldRun = Ref(true)

function update(g, shitload, avgs, img_ob)
    t = Threads.@spawn begin
        updateshitload(shitload, state(g[1]))
        updateavgs(avgs, shitload)
    end
    wait(t)
    notify(img_ob)
end

timer = Timer(t -> update(g, shitload, avgs, img_ob), 0, interval = 1/1024)
