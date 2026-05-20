# MNIST Interactive Desktop App

This folder contains a desktop-app entrypoint for the GLMakie MNIST demo. It is
kept outside the core `InteractiveIsing` module because it depends on the
`ext/IsingLearning` project, `MLDatasets`, and the trained MNIST checkpoint.

Run it from the repository root with four Julia threads:

```sh
julia --project=ext/IsingLearning -t 4 src/App/run_mnist_interactive_app.jl
```

Run the non-window smoke check with:

```sh
julia --project=ext/IsingLearning -t 4 src/App/run_mnist_interactive_app.jl --check
```

The module exposes `MNISTInteractiveApp.julia_main()`, so it can be used as the
entrypoint for a later `PackageCompiler.create_app` build.
