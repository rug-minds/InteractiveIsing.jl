using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using IsingLearning, IsingLearning.InteractiveIsing, IsingLearning.InteractiveIsing.Processes, Lux, Random, SparseArrays

# MNIST TEST
rbm = ReducedBoltzmannArchitecture(784, 100, 10; precision = Float64)

rbm2 = GraphFromSource(rbm)

lux_model = LayeredIsingGraphLayer(() -> ReducedBoltzmannArchitecture(784, 100, 10; precision = Float64); input_idxs = layerrange(rbm2[1]), output_idxs = layerrange(rbm2[end]))
ps, st = Lux.setup(Random.default_rng(), lux_model)

dynamics = NudgedProcess(lux_model)

buffers = (;w = zeros(length(SparseArrays.getnzval(adj(rbm)))), b = zeros(nstates(rbm)), α = zeros(nstates(rbm)))
algo = dynamics.algorithm
reg = getregistry(algo)
reg
p = InlineProcess(dynamics.algorithm, Input(:_state; buffers), Input(:dynamics, state = rbm), Input(:plus_capture, state = rbm), Input(:minus_capture, state = rbm); repeats = 1)
rc = run(p)


test(rc, rbm) = Processes.initcontext(rc, :dynamics, inputs = (;state = rbm))
rc = Processes.initcontext(rc, :dynamics, inputs = (;state = rbm))

@code_warntype test(rc, rbm)