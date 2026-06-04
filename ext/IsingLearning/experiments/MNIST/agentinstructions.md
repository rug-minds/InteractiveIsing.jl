# MNIST Experiment Instructions

Use of this file: operational rules for future agents before running or editing MNIST experiments. Keep this as instructions, not results.

- Do diagnostic runs before broad grids. Poll periodically and stop if accuracy or loss is clearly not improving.
- Do not resume from a checkpoint unless the user explicitly asks for a resumed/continued/fine-tuning run, or the experiment design says checkpoint initialization is the variable being tested. If resuming, write the exact checkpoint path in the launcher and settings before starting the run.
- Use `ProcessManager` for manager-backed MNIST training experiments. Start Julia with enough threads, normally `julia -t 32 --project=ext/IsingLearning ...`.
- Do not use `ProcessManager` for interactive runtime demos. Interactive MNIST should be a continuous single-graph simulation, normally one `LocalLangevin` loop, with no training manager.
- Reuse worker contexts after creation. Do not rebuild processes, graphs, or contexts per batch.
- Use public `StatefulAlgorithms`/`InteractiveIsing` APIs from experiments. Do not reach into `StatefulAlgorithms` internals unless a confirmed package bug blocks the run.
- Do not use `createProcess` or graph-input merge helpers in MNIST training code. If the experiment authors the `LoopAlgorithm`, provide the needed `Init`s directly.
- A minibatch is a minibatch, not an epoch. An epoch is the whole selected training split. Let each worker accumulate several examples into its own buffer, then flush/sync once after the minibatch.
- Keep at least as many jobs as useful workers when benchmarking parallelism. For training, choose batch/chunking so worker jobs contain enough examples to amortize scheduling and still fill the workers.
- Share static model data across workers where possible, especially `J`/adjacency and base parameter arrays. Worker state and clamp buffers stay worker-local.
- If using field-based image input, keep the base model parameters shared and write only worker-local input-field buffers between samples. Do not mutate the source graph state just to load an image.
- For the `784-120-40` field-input baseline, do not keep a structural input layer in the sampled graph. Keep the `784 -> hidden` weights as an external projection into a worker-local magnetic field, and sample only hidden/output units.
- Treat fixed input spins and image-dependent magnetic fields as equivalent input mechanisms. Use the field path when it avoids graph-state writes and keeps worker data sharing intact.
- Count relaxation settings in full sweeps when discussing learning. Raw step counts like `300` are not enough for a large MNIST graph unless explicitly converted from sweeps.
- Use Adam by default for current MNIST runs. Do not assume Muon is applicable to sparse coupling-vector parameters without a deliberate implementation.
- Do not use vague `paper_*` labels in new run names, legends, or documentation. Describe the actual architecture and parameters instead.
- Do not use implementation details such as `input-field` in architecture folder names. Folder names should describe the result or architecture, for example hidden/output sizes; implementation details can go in code comments or run settings.
- Do not auto-generate architecture-folder README files. Keep durable process rules in this `agentinstructions.md`; write run-specific settings files only when an experiment run needs reproducibility metadata.
- The single-hidden local baseline should be kept available: `784 -> about 120 hidden -> 40 output` with four output replicas per digit. Use it as the first sanity check before larger local architectures.
- The larger local/CNN-like MNIST architecture should live in its own architecture folder. Use `two-hidden-local-28x28-to-14x14-readout/mnist_cnn_two_layer_nn_grid.jl` for the two-hidden-layer CNN-style manager runs and compare local radius/NN only after one case shows sane learning and timing.
- For the inlaid-input architecture, keep the 28x28 MNIST pixels fixed inside the 55x55 input layer and sample only separator sites plus output sites. Do not toggle the whole input layer off.
- Before training the inlaid-input architecture, run `mnist_inlaid_input_diagnostics.jl` with `-t 32` and confirm pixel states stay fixed, relaxation is in the 75-100 full-sweep range, and 32-worker throughput scaling is still sane.
- For `mnist_inlaid_input_training.jl`, keep fixed pixel values in `[0, 1]`, not bipolar `[-1, 1]`, unless explicitly testing that encoding. Use a separate `applied_bias_clip` for target fields so β is not accidentally clipped to the trainable bias range.
- When adapting the one-sided contrastive formula to `Optimisers.jl`, remember that the legacy manager buffer was an update direction for manual `+=`; negate it before Adam/descent.
- Current CNN-style MNIST recipes are sensitive to hidden2 size: `11x11` worked with the 16x mean-normalized LR scale, while `14x14` needed the 8x scale.
- Track batch time, epoch time, train accuracy, test accuracy, train loss, test loss, prediction counts, and skipped samples if any.
- Save best parameters, final parameters, metrics CSVs, plots, and the exact run settings for any run that learns.
- After a run succeeds, write `success.md` in that experiment run folder. Use it to explain what made the run work: architecture choices, optimizer settings, input encoding, relaxation settings, batching/jobs, validation checks, and any failed assumptions that were corrected. Keep the architecture `README.md` focused on experiment motivation and design rather than run-result dumps.
- If a run writes CSVs, every CSV needs a matching PNG in the same `experiments/current/<experiment-name>` folder, or a deliberate aggregate PNG under that architecture folder's `aggregate_plots`. Checkpoint, post-hoc eval, summary, and diagnostic CSVs also need plots, not just training CSVs.
- Every aggregate MNIST architecture family needs a generated `schematic.png` directly in the architecture folder. Regenerate it with `ext/IsingLearning/experiments/plot_architecture_schematics.jl` after adding a new architecture family.
- Do not recreate `runs`, `schematics`, or `individual_plots` under `MNIST`. Current learning experiments belong under `experiments/current/<experiment-name>` inside the relevant architecture folder; diagnostics belong under `MNIST/diagnostics/runs`.
- Aggregate plots must use line styles and stroke widths as well as colors. Current convention: NN/radius 1-3 solid/thinner, 4-7 dashed/medium, 8+ dotted/thicker.
- Do not run large grids until a short diagnostic confirms the recipe is learning and the timing is sane.
- Do not hide startup or package issues with weird command-line workarounds such as `--startup-file=no`; fix or document the actual cause.
- Update this file whenever a structural experiment mistake is found.
