using Dates
function logline(message)
    println(Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), " ", message)
    flush(stdout)
end
logline("probe start")
include(raw"C:/Users/fenje/dev/InteractiveIsing.jl/ext/IsingLearning/experiments/MNIST/784-120-40-baseline/diagnostics/runs/20260531_main_codegen_check/temp_root/ext/IsingLearning/experiments/MNIST/784-120-40-baseline/mnist_784_120_40_adam.jl")
logline("baseline included looptype=$(IsingLearning.InteractiveIsing.Processes.sys_looptype)")
config = InputFieldMNISTConfig(; workers=1, epochs=1, batchsize=1, train_per_class=2, test_per_class=1, train_eval_per_class=0, eval_every=1, sweeps=500f0, outdir=raw"C:/Users/fenje/dev/InteractiveIsing.jl/ext/IsingLearning/experiments/MNIST/784-120-40-baseline/diagnostics/runs/20260531_main_codegen_check")
logline("load data")
xtrain, ytrain = balanced_mnist(:train, config.train_per_class, config)
logline("build layer")
setup = build_layer(config)
source_graph = setup.graph
input_hidden_w_ref = Ref(copy(setup.input_hidden_w))
algorithm = Processes.resolve(input_field_contrastive_algorithm(setup.layer))
worker = input_field_worker(algorithm, setup.layer, shared_worker_graph(source_graph), input_hidden_w_ref)
try
    load_sample_into_worker!(worker_context(worker), xtrain, ytrain, 1)
    Processes.reset!(worker)
    inputs = Processes._validate_runtime_inputs(algorithm, (; phase_beta = config.β))
    base_context = Processes._has_typed_runtime_context(worker) ? Processes._typed_runtime_context(worker) : Processes.context(worker)
    lt = Processes._has_typed_runtime_context(worker) ? Processes._context_lifetime(base_context) : get(Processes.getglobals(base_context), :lifetime, Processes.Indefinite())
    logline("explicit generated loop begin context_type=$(typeof(base_context))")
    wall = @elapsed Processes.loop(worker, algorithm, base_context, lt, inputs, Processes.Resuming{false}(), Processes.Generated())
    logline("explicit generated loop end wall=$wall")
finally
    close(worker)
    logline("probe closed")
end
