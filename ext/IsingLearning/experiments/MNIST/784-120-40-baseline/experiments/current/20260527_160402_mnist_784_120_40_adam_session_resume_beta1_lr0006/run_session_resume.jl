include(raw"C:/Users/fenje/dev/InteractiveIsing.jl/ext/IsingLearning/experiments/MNIST/784-120-40-baseline/mnist_784_120_40_adam.jl")
config = InputFieldMNISTConfig(
    workers = 32,
    epochs = 300,
    β = FT(1.0),
    lr = FT(0.0006),
    outdir = raw"C:/Users/fenje/dev/InteractiveIsing.jl/ext/IsingLearning/experiments/MNIST/784-120-40-baseline/experiments/current/20260527_160402_mnist_784_120_40_adam_session_resume_beta1_lr0006",
)
session = create_session(config)
try
    start_epoch = restart_from_checkpoint!(
        session,
        raw"C:/Users/fenje/dev/InteractiveIsing.jl/ext/IsingLearning/experiments/MNIST/784-120-40-baseline/experiments/current/20260527_094101_mnist_784_120_40_adam_balanced_e300/best_checkpoint.bin";
        resume_epoch = 16,
        beta = FT(1.0),
        lr = FT(0.0006),
        outdir = raw"C:/Users/fenje/dev/InteractiveIsing.jl/ext/IsingLearning/experiments/MNIST/784-120-40-baseline/experiments/current/20260527_160402_mnist_784_120_40_adam_session_resume_beta1_lr0006",
    )
    run_epochs!(session; start_epoch = start_epoch, stop_epoch = session.config.epochs)
    finalize_session!(session)
finally
    close_session!(session)
end
