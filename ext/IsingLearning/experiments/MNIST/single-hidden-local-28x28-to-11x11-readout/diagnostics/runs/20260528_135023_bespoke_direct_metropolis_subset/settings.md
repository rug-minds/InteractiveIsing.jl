# Direct Metropolis subset timing

- created: `2026-05-28T16:52:53`
- manager source: `C:\Users\fenje\dev\InteractiveIsing.jl\ext\IsingLearning\experiments\MNIST\single-hidden-local-28x28-to-11x11-readout\mnist_local_manager_grid.jl`
- ProcessManager: `false`
- dynamics: manual direct Metropolis over `II.adj(model.graph)` CSC storage
- radius: `8`
- free/nudge sweeps: `50` / `50`
- free/nudge reads: `3` / `3`
- batch size: `32`
- measured subset samples: `2` / `1000`
- subset default: `ceil(full_epoch_samples * 60 / 220)`
