using Dates
function logline(message)
    println(Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), " ", message)
    flush(stdout)
end
logline("probe start")
include(raw"C:/Users/fenje/dev/InteractiveIsing.jl/ext/IsingLearning/experiments/MNIST/784-120-40-baseline/diagnostics/runs/20260531_main_codegen_check/temp_root/ext/IsingLearning/experiments/MNIST/784-120-40-baseline/mnist_784_120_40_adam.jl")
logline("baseline included looptype=$(IsingLearning.InteractiveIsing.StatefulAlgorithms.sys_looptype)")
config = InputFieldMNISTConfig(; workers=1, epochs=1, batchsize=1, train_per_class=2, test_per_class=1, train_eval_per_class=0, eval_every=1, sweeps=500f0, outdir=raw"C:/Users/fenje/dev/InteractiveIsing.jl/ext/IsingLearning/experiments/MNIST/784-120-40-baseline/diagnostics/runs/20260531_main_codegen_check")
logline("load data")
xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)
logline("build layer")
setup = build_layer(config)
source_graph = setup.graph
input_hidden_w_ref = Ref(copy(setup.input_hidden_w))
algorithm = StatefulAlgorithms.resolve(input_field_contrastive_algorithm(setup.layer))
worker = input_field_worker(algorithm, setup.layer, shared_worker_graph(source_graph), input_hidden_w_ref)
try
    load_sample_into_worker!(worker_context(worker), xtrain, ytrain, 1)
    StatefulAlgorithms.reset!(worker)
    inputs = StatefulAlgorithms._validate_runtime_inputs(algorithm, (; phase_beta = config.β))
    base_context = StatefulAlgorithms._has_typed_runtime_context(worker) ? StatefulAlgorithms._typed_runtime_context(worker) : StatefulAlgorithms.context(worker)
    lt = StatefulAlgorithms._has_typed_runtime_context(worker) ? StatefulAlgorithms._context_lifetime(base_context) : get(StatefulAlgorithms.getglobals(base_context), :lifetime, StatefulAlgorithms.Indefinite())
    logline("explicit generated loop begin context_type=$(typeof(base_context))")
    wall = @elapsed StatefulAlgorithms.loop(worker, algorithm, base_context, lt, inputs, StatefulAlgorithms.Resuming{false}(), StatefulAlgorithms.Generated())
    logline("explicit generated loop end wall=$wall")
finally
    close(worker)
    logline("probe closed")
end
