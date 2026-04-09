 using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using IsingLearning, IsingLearning.InteractiveIsing, IsingLearning.InteractiveIsing.Processes, Lux, Random, SparseArrays
using BenchmarkTools

# MNIST TEST
rbm = ReducedBoltzmannArchitecture(784, 100, 10; precision = Float64)

rbm2 = GraphFromSource(rbm)

lux_model = LayeredIsingGraphLayer(() -> ReducedBoltzmannArchitecture(784, 100, 10; precision = Float64); input_idxs = layerrange(rbm2[1]), output_idxs = layerrange(rbm2[end]), β = 1.0, fullsweeps = 10)
ps, st = Lux.setup(Random.default_rng(), lux_model)

dynamics = NudgedDynamics(lux_model)

xmock = rand(length(layerrange(rbm2[1])))
ymock = rand(length(layerrange(rbm2[end])))

buffers = (;w = zeros(length(SparseArrays.getnzval(adj(rbm)))), b = zeros(nstates(rbm)), α = zeros(nstates(rbm)))
algo = dynamics.algorithm
reg = getregistry(algo)
reg
p = InlineProcess(dynamics.algorithm, Input(:_state; buffers, equilibrium_state = copy(state(rbm)), x = xmock, y = ymock),
    Input(:minus_capture, state = rbm),
    Input(:dynamics, state = rbm); repeats = 1)
rc = run(p)
@benchmark run(p)

test(rc, rbm) = Processes.initcontext(rc, :dynamics, inputs = (;state = rbm))
rc = Processes.initcontext(rc, :dynamics, inputs = (;state = rbm))

@code_warntype test(rc, rbm)

full_process = Forward_and_Nudged(lux_model)

fp = resolve(full_process.algorithm)
p = InlineProcess(fp, Input(:_state; buffers, equilibrium_state = copy(state(rbm)), x = xmock, y = ymock), 
                    Input(:dynamics, state = rbm), Input(:plus_capture, state = rbm), 
                    Input(:minus_capture, state = rbm); repeats = 1)
rc = run(p)
@benchmark run(p)
@code_warntype run(p)