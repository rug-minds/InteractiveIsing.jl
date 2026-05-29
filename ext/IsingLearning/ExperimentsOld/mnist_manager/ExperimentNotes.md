# MNIST Manager Experiment Notes

## 2026-05-20 Startup Profiling

Benchmark target for comparison:

- architecture: `28^2 -> 120 -> 10`
- workers: `15,16`
- batch size: `32`
- relaxation: `1.0` sweep over active hidden/output units, currently `130` single-spin steps
- Julia launched with `--project=ext/IsingLearning -t 16` and normal startup file

Findings so far:

- `Pkg.activate(...)` inside `Profiling.jl` is not the main delay. With `--project=ext/IsingLearning`, loading `Pkg` and activating the same project cost about `0.43s`.
- `using IsingLearning` after precompile is about `7s` in a fresh Julia process.
- The large pre-row delay is worker/process construction, not import or MNIST dataset loading.
- Drilldown for a 16-worker setup:
  - `build graph`: about `2.12s`
  - `build layer`: about `0.00s`
  - `read params`: about `0.07s`
  - `one worker graph`: about `0.34s`
  - `one worker process`: about `18.12s`
  - `copy worker process`: about `8.17s`
  - full manager constructor after that: about `19.48s`
- The repeated warnings about overlapping `GeneralState` fields appear during process construction:
  - `equilibrium_state`, `x`
  - `equilibrium_state`, `y`, `x`

Working hypothesis:

- `Process` construction for the MNIST `Forward_and_Nudged` loop is the expensive step.
- The default `ProcessManager` path copies a template process/context for owned workers. That may be expensive because each context contains graph-backed dynamics state and capture buffers.
- Next check: compare repeated fresh worker construction against template-context copying after the first compile hit.

Follow-up measurement:

- Fresh `_worker_process(layer, graph)` remained very slow even after warmup:
  - warmup worker process: about `21.7s`
  - repeated fresh worker processes: about `14.6s`, then `24s` range
- Context-copying an already initialized process was fast after the first copy:
  - first copy: about `0.59s`
  - later copies: about `0.001s`
- Fresh worker graph creation after warmup was essentially free at this size, about `0.0006s`.

Updated hypothesis:

- The expensive part is not copying worker contexts. It is initializing a fresh `Process` for the composed `Forward_and_Nudged` loop.
- The next split is `Forward_and_Nudged(layer)`, `resolve(...)`, and `Process(resolved_algo, Init(...))`.

Algorithm construction split:

- `Forward_and_Nudged(layer).algorithm` costs about `6.7s` to `8.8s` every time it is built.
- `resolve(deepcopy(raw_forward_and_nudged))` costs about `12.2s` the first time, then about `0.14s`.
- `Process(resolved_algo, inputs...)` costs about `1.7s` the first time, then about `0.0002s`.
- `Process(unresolved_algo, inputs...)` remains about `8.3s` per call because it forces algorithm construction/resolution internally.
- Forward-only validation is cheaper but still has a first resolve hit: `1.42s` first, then about `0.038s`.

Action taken:

- Added a dedicated `MNISTContrastiveStep` process algorithm in `ThreadedMNISTLoop.jl`.
- This avoids rebuilding the nested `Forward_and_Nudged(layer).algorithm` routine for each worker.
- The worker still uses an explicit `Process` and explicit `Init`; no `createProcess` or graph-input merge helper is used by the learning code.

Custom worker smoke test:

- After the edit precompiled, a single 120-hidden worker constructs in about `3.2s` from the timestamped probe.
- One process run with `free_relaxation_steps=2`, `nudged_relaxation_steps=2` takes about `0.85s` including first execution compile in the fresh Julia process.
- The run produced a nonzero worker gradient buffer (`sum(abs, buffer.w) ~= 5.7` on the second probe), so the custom step is executing the three phases and accumulating into the persistent buffer.
- The remaining startup floor for a fresh Julia process is roughly `6-7s` for `using IsingLearning`, plus graph/layer construction.

First custom-step profiling pass:

- Same comparison target as above: hidden `120`, batch `32`, workers `15,16`, one warmup batch and one measured batch.
- The run completed after fixing one remaining old-context lookup in `init_mnist_trainer`.
- `build_trainer_seconds` dropped from the old explicit-init baseline of about `34s` for 15 workers and `22s` for 16 workers to:
  - 15 workers: about `10.05s`
  - 16 workers: about `5.37s`
- Measured post-warmup batch timings remained in the same range:
  - 15 workers: `manager_run ~= 0.00318s`, `collect_gradient ~= 0.00096s`, `serial_sync ~= 0.00075s`
  - 16 workers: `manager_run ~= 0.00373s`, `collect_gradient ~= 0.00089s`, `serial_sync ~= 0.00078s`
- The striped sync prototype is currently slower (`0.017-0.032s`) than direct serial/graph-threaded sync for this parameter size. It is correct (`striped_sync_error == 0`) but not worth using at hidden `120`.
- This pass still ended with package precompile because `ThreadedMNISTLoop.jl` had just changed; rerun once after precompile for cleaner startup/build numbers.

Clean custom-step profiling rerun:

- Output directory: `runs/20260520_profile_custom_step_clean`.
- 15 workers:
  - `build_graph_seconds ~= 2.07`
  - `build_trainer_seconds ~= 10.00`
  - measured batch: `manager_run ~= 0.00346s`, `collect_gradient ~= 0.00086s`, `serial_sync ~= 0.00084s`
- 16 workers:
  - `build_graph_seconds ~= 0.044` after first graph compile had already happened
  - `build_trainer_seconds ~= 5.26`
  - measured batch: `manager_run ~= 0.00350s`, `collect_gradient ~= 0.00095s`, `serial_sync ~= 0.00091s`
- The custom `MNISTContrastiveStep` removes the worst startup issue. There is still a 5-10s manager construction cost for 15-16 workers, but it is no longer the 20-35s delay from repeatedly constructing `Forward_and_Nudged`.
- At hidden `120`, direct graph-threaded/serial sync is faster than the striped prototype. The stripe idea may only become relevant for much larger parameter arrays.

## 2026-05-20 120-Hidden Learning Probes

Paper architecture note:

- The referenced Nature Communications paper uses a much smaller hidden layer than the earlier `10 * MNIST` setup: about `120` hidden units.
- It also uses `40` output units for MNIST, i.e. four output spins per digit class. The practical reason is that a class score can be represented by a group of spins rather than one noisy binary/continuous spin.
- I added default-preserving support for this in the MNIST learning path:
  - `MNISTArchitecture(output_replicas = 4)` gives `784 -> hidden -> 40`.
  - `load_mnist_arrays` now emits repeated class targets when the output layer is `10 * replicas`.
  - MNIST evaluation and forward-manager prediction reduce output replicas by summing each class group.
  - The default remains `output_replicas = 1`, so existing 10-output demos keep their shape.

Small training results:

- `784 -> 120 -> 10`, 16 workers, 50 sweeps, 256 train examples:
  - One epoch moved train MSE `6.31 -> 5.23`, but validation stayed around chance.
- `784 -> 120 -> 40`, 16 workers, 50 sweeps, 256 train examples:
  - Validation accuracy moved from about `14%` to `39%` after two small epochs.
  - This is the first clearly positive MNIST learning signal in these runs.

Manager stability finding:

- With `16` workers on `16` Julia threads, longer multi-worker runs can hard-exit with no Julia stack trace.
- The same workload with `1` worker completes, and a `15` worker run completes for smaller/more stable dispatch shapes.
- Leaving one thread free (`15` workers on `16` threads) appears more stable, matching the earlier CPU-utilization suspicion.
- The hard exit also depends on dispatch granularity: `1024` examples with batch size `32` hard-exited after one epoch, while batch size `64` completed.
- Working default for now: `15` workers, batch size `64`, 50 sweeps.

Saved run:

- Directory: `runs/20260520_784_120_40_15w_lr001_b64_10epoch_saved`.
- Settings: hidden `120`, output replicas `4`, workers `15`, train limit `1024`, validation limit `128`, batch size `64`, 50 sweeps, `lr=0.001`, `β=0.1`, `T=0.001`.
- Best checkpoint by validation MSE:
  - epoch `7`
  - train accuracy `31.25%`
  - validation accuracy `37.5%`
  - validation MSE `18.34`
- Checkpoints written:
  - `checkpoints/config_1/initial_graph.jld2`
  - `checkpoints/config_1/best_graph.jld2`
  - `checkpoints/config_1/final_graph.jld2`

Current diagnosis:

- The system is learning something with the 40-output architecture, but accuracy is noisy and can regress after a good epoch.
- The next useful probes are lower learning rate with the same batch-64 setup, and possibly multiple initial-state averages per sample once the manager stability issue is understood.

## 2026-05-20 Manager API Audit

Public API use:

- The MNIST training path now builds workers with the custom `MNISTContrastiveStep` directly. It does not use `createProcess` or the graph-input injection helper.
- The manager recipe uses the public `WorkerSlot.worker` field, `resetworker!`, `NoFlush`, and `run!`.
- `WorkerSlot.worker` is documented in `Processes` as intentionally public for recipes that mutate worker context directly.
- `reset!(::Process)` only resets process lifecycle counters/timing/algorithm state. It does not rebuild the context, so writing `x` and `y` into the existing `:_state` context and then calling `resetworker!` reuses the worker context.
- The worker process has one named subcontext, `:_state`, containing the model graph, input vector, target vector, persistent gradient buffers, and phase state arrays.

Context reuse checks:

- A 4-worker construction probe produced `unique_graphs=4` and `unique_buffers=4`, so manager-owned workers have separate graph objects and separate buffer arrays.
- The same probe saw `state_key_y=40` for `output_replicas=4`, confirming the worker context uses the repeated-output target shape.
- A smoke run with two relaxation steps produced a nonzero worker buffer (`buffer_abs_sum ~= 114.26` in the most recent probe), so the custom process executes and accumulates into its persistent buffer.

Working experiment defaults:

- Use `-t 16` and `15` manager workers for now. The 16-worker case can still hard-exit on longer runs, likely because every Julia thread is occupied by worker tasks and the manager/runtime has no spare execution capacity.
- Use batch size `64` for the 1024-example probes. It has fewer dispatch waves than batch size `32` and has been more stable in the current manager setup.
- Keep direct serial/graph-threaded parameter sync at hidden `120`; the striped sync prototype was correct but slower for this parameter size.

Follow-up profiling after cleaning the experiment script:

- I fixed `Profiling.jl` to use `output_replicas=4`, typed graph lists for sync timing, and a concrete vector of job named-tuples. The earlier `Vector{Any}` in the profiling file was an experiment artifact, not part of `fit_mnist_threaded!`.
- Short profile settings: hidden `120`, output replicas `4`, batch size `64`, one warmup batch, three measured batches, 5 sweeps (`800` relaxation steps).
- Average measured batch timings:
  - 8 workers: manager `15.01ms`, collect gradient `0.38ms`, broadcast params `0.52ms`.
  - 12 workers: manager `11.53ms`, collect gradient `0.50ms`, broadcast params `0.71ms`.
  - 15 workers: manager `9.70ms`, collect gradient `0.64ms`, broadcast params `0.84ms`.
  - 16 workers: manager `9.77ms`, collect gradient `1.04ms`, broadcast params `0.81ms`.
- Worker count scaling plateaus around 15 workers for this short-job profile. 16 workers is not materially faster and has been less stable on longer runs.
- Typed striped sync and graph-threaded sync are now both about `0.5-0.6ms` at 15-16 workers. The production `_broadcast_params!` now syncs worker graphs in parallel after each minibatch; a focused 15-worker profile measured it around `0.56-0.58ms`.

Training smoke after manager/sync audit:

- Directory: `runs/20260520_manager_audit_15w_b64_parallel_sync`.
- Settings: hidden `120`, output replicas `4`, 15 workers on 16 Julia threads, batch size `64`, train limit `1024`, validation limit `128`, 50 sweeps (`8000` relaxation steps), `lr=0.001`, `β=0.1`, `T=0.001`.
- The first run after editing `ThreadedMNISTLoop.jl` spent about `95s` precompiling `IsingLearning`; after that, epoch times were about `0.92-0.98s`.
- The run completed and checkpointed. Validation accuracy stayed above chance but was noisier than the earlier run:
  - initial validation accuracy `9.4%`
  - best validation MSE at epoch `1`, validation accuracy `23.4%`
  - epoch `5` validation accuracy `29.7%`
- This confirms the manager path still runs correctly after parallel worker-graph sync, but it also confirms that the learning recipe is still noisy and needs parameter tuning rather than more manager surgery.

## 2026-05-20 Relaxation Response Diagnostics

Paper anchor:

- Laydevant, Markovic, and Grollier, "Training an Ising machine with equilibrium propagation", Nature Communications 15, 3671 (2024), use MNIST with `784 x 120` input-to-hidden software biasing and an embedded `120 -> 40` Ising machine readout.
- Their 40 outputs are four spins per digit class. The paper explicitly reports that this makes the nudge phase function effectively.
- Their D-Wave run uses annealing/reverse annealing rather than our LocalLangevin dynamics, so the paper is evidence that the architecture and EP objective can work, not evidence for our specific step count.

New diagnostic file:

- Added `mnist_relaxation_response.jl`.
- It measures two curves before training:
  - free phase: distance of checkpoints to the longest same-sample free trajectory;
  - nudged phase: plus/minus movement from the free state, including output loss deltas and target-direction alignment.
- Defaults match the paper-inspired architecture: hidden `120`, output replicas `4`, target scale `0.2`, `T=0.001`, `weight_scale=0.005`.

Response grid:

- Directory: `runs/20260520_response_grid_local_langevin`.
- Settings: 32 samples, LocalLangevin, hidden `120`, output `40`, checkpoints `1,2,5,10,20,50,100` active-layer sweeps, stepsizes `0.05,0.1,0.2,0.5`, betas `0.05,0.1,0.2`.
- Free response:
  - `stepsize=0.05` gave the best free output MSE at the 100-sweep reference (`~16`).
  - `stepsize=0.1` was worse (`~19`).
  - `stepsize=0.2` and `0.5` moved more aggressively but produced much worse free output states (`~24-34` MSE).
- Nudged response:
  - Even at `stepsize=0.05`, plus nudging reliably reduced output loss and minus nudging increased it.
  - `beta=0.2` produced a stronger aligned response than `beta=0.1`, but later short training showed it hurt classification stability.

Long response pass:

- Directory: `runs/20260520_response_long_local_langevin`.
- Settings: 16 samples, checkpoints `20,50,100,200,300` active-layer sweeps.
- For `stepsize=0.05`, `beta=0.1`, free output RMS distance to the 300-sweep reference was:
  - 20 sweeps: `~0.454`
  - 50 sweeps: `~0.415`
  - 100 sweeps: `~0.344`
  - 200 sweeps: `~0.177`
  - 300 sweeps: reference
- This says the earlier 50-sweep runs were under-relaxed for the free phase. A reasonable current free-phase setting is around `200` active-layer sweeps, with `300` being safer but slower.
- For the same config, plus nudged output movement was already clear:
  - 50 nudged sweeps: plus loss delta `~-4.90`
  - 100 nudged sweeps: `~-7.80`
  - 200 nudged sweeps: `~-10.46`
- Current educated guess: use `free_sweeps=200`, `nudged_sweeps=100`, `stepsize=0.05`, `beta=0.1` as the first serious LocalLangevin training point.

Training checks using separate free/nudged sweeps:

- Added `ISING_MNIST_SMALL_FREE_SWEEPS` and `ISING_MNIST_SMALL_NUDGED_SWEEPS` to `mnist_small_sweep_grid.jl`; defaults fall back to `ISING_MNIST_SMALL_SWEEPS`.
- Short 3-epoch runs, 1024 train / 128 validation, 15 workers, batch 64:
  - `free=100`, `nudged=50`, `beta=0.1`: best validation accuracy `27.3%`, best validation MSE `15.33`.
  - `free=100`, `nudged=100`, `beta=0.1`: best validation accuracy `31.3%`, best validation MSE `14.34`.
  - `free=200`, `nudged=50`, `beta=0.1`: best validation accuracy `25.8%`, best validation MSE `15.55`.
  - `free=200`, `nudged=100`, `beta=0.1`: best validation accuracy `41.4%`, best validation MSE `16.04`.
  - `free=300`, `nudged=100`, `beta=0.1`: best validation accuracy `28.1%`, best validation MSE `17.19`.
- `beta=0.2` with `free=200` reduced MSE more, but classification was worse:
  - `nudged=50`: best validation accuracy `30.5%`, best validation MSE `14.02`.
  - `nudged=100`: best validation accuracy `22.7%`, best validation MSE `14.30`.

Extended runs:

- `free=200`, `nudged=100`, `beta=0.1`, `lr=0.001`, 10 epochs:
  - best validation MSE `14.17` at epoch `9`;
  - best validation accuracy in that run `32.8%` at epoch `1`, not the `41.4%` seen in the 3-epoch run.
- Same but `lr=0.0005`, 10 epochs:
  - best validation MSE `13.59` at epoch `7`;
  - validation accuracy remained noisy and mostly `15-29%`.

Current interpretation:

- The graph response diagnostics do justify increasing relaxation from 50 sweeps to roughly `free=200`, `nudged=100` for LocalLangevin.
- The model is learning something by MSE, but classification accuracy is still unstable. The next likely issue is the readout/target dynamics rather than manager plumbing or simple lack of relaxation.
- Candidate next probes: repeated initial states per example (`MINIT > 1`), output time averaging, and possibly a smaller beta with more nudged averaging. Metropolis remains a fallback if LocalLangevin response remains too noisy.

## 2026-05-20 Balanced MNIST and Warm-Start Diagnostics

New experiment files:

- `mnist_timeavg_search.jl`: local `ProcessAlgorithm` that averages multiple nudged samples before accumulating the symmetric EP gradient.
- `mnist_balanced_grid.jl`: stock `MNISTContrastiveStep` trainer, but with balanced train/validation subsets and grid controls for target coding, beta, LR, and explicit gradient-sign checks.
- `mnist_hidden_ridge_warmstart.jl`: measures relaxed hidden states, fits only the hidden-to-output readout by ridge regression, installs that readout in the graph, and can continue with the stock EP manager loop.

Balanced-target finding:

- The old 128-example validation slices were too noisy for tuning; the first 128 test examples contain only three `8`s.
- Balanced 500-train / 200-validation grids with the stock random graph stayed near chance:
  - best `target_off=-target_on` result: about `13.5%` validation accuracy.
  - best `target_off=0` result: about `13.0%` validation accuracy.
- Flipping the accumulated gradient sign made MSE blow up to about `40`, so the source gradient sign is not simply reversed.
- The stock random graph does reduce MSE, but on balanced validation it does not learn class-selective outputs reliably.

Hidden-feature warm start:

- With 120 hidden units and four output replicas, a measured-hidden-state ridge readout gives a useful baseline.
- Settings for the best current probe:
  - hidden `120`, output replicas `4`.
  - balanced train `100/class`, validation `50/class`.
  - `weight_scale=0.05`, `ridge=10.0`, `target_on=0.5`, `target_off=0.0`.
  - LocalLangevin `stepsize=0.05`, `free_sweeps=50`, `nudged_sweeps=25`, `T=0.001`.
- Ridge-only validation:
  - random graph before ridge: around `7-12%`.
  - after hidden-state ridge: best observed balanced validation accuracy `40.4%` in the lambda sweep; repeated run was `35.6%`.
- EP continuation from the ridge warm start:
  - `lr=1e-4`, three epochs: ridge `35.2%`, EP epochs `31.0%`, `33.4%`, `34.6%`.
  - `lr=1e-5`, ten epochs: ridge `35.6%`, best EP validation accuracy `37.4%` at epoch 4, best MSE `13.94` at epoch 6.
- Interpretation:
  - The 120-hidden graph can carry digit information, but the fully random EP run does not discover a good readout from scratch in these short balanced runs.
  - Larger initial input-hidden/readout scale is important. `weight_scale=0.05` was much better than `0.005`.
  - A fitted readout gives a real 120-hidden/40-output baseline; stock EP can be stable at a very small LR but has not yet clearly improved beyond the warm start.

## 2026-05-20 Paper-Like Ising EP Translation

Paper/code audit:

- The released Laydevant code does not embed MNIST pixels as Ising spins. It uses `784 -> 120` as an external input-to-hidden bias map, then samples only a `120 -> 40` hidden/output Ising model.
- The 40 outputs are four repeated spins per digit. The target is `-1` for non-class outputs and `+1` for the class replicas.
- The nudge phase is not the quadratic clamp used in the earlier local experiments. In Ising mode the released code shifts output fields by `-beta * target` and then samples a reverse-annealed nudged state from the free state.
- The learning rule is one-sided EP: update from `nudged correlations - free correlations`, not symmetric plus/minus clamping.
- Important sign mismatch found: `dimod` uses Ising energy `h*s + J*s*s`, while this package's `Bilinear + MagField` uses `-b*s - J*s*s`. The paper parameters must be installed into the graph with a minus sign for both fields and couplings.

New experiment file:

- Added `mnist_paper_like_ep.jl` as a self-contained paper-style learner.
- It keeps the paper parameters (`weights_0`, `weights_1`, `bias_0`, `bias_1`) separate from the graph. The graph only receives the negated hidden-output couplings and current per-sample fields.
- It serializes `best_model.bin` and `final_model.bin` in the run directory.

Smoke/pilot results:

- Tiny smoke run completed and wrote checkpoints after replacing unsupported full-H energy with an explicit hidden-output energy.
- A first translation mistake used only `free_hidden * output_delta` for the hidden-output update; the paper code uses the full correlation difference `nudged_hidden*nudged_output' - free_hidden*free_output'`.
- Before fixing the Hamiltonian sign, 10/class runs collapsed to a single predicted class after one epoch.
- After fixing the sign convention, run `runs/20260520_paper_like_10pc_signed_r3_s20` learned clearly:
  - settings: train `10/class`, test `5/class`, reads `3/3`, sweeps `20/20`, epochs `5`, beta `5`.
  - test accuracy: `12%` at initialization, then `16%`, `42%`, `44%`, `44%`, `54%`.
  - test loss: `41.08` at initialization to `7.60` by epoch 5.

Current interpretation:

- The earlier failures were not just parameter tuning. The previous graph formulation trained the wrong Hamiltonian sign relative to the paper, and it also used a different nudge mechanism.
- With paper-style fields, output replicas, one-sided EP, and the correct energy sign, MNIST starts learning even with very short simulated annealing.

Paper-style training runs:

- Run `runs/20260520_paper_like_100pc_r3_s50_e15`:
  - settings: train `100/class`, test `10/class`, reads `3/3`, sweeps `50/50`, epochs `15`, beta `5`.
  - best test accuracy `89%` at epoch 14; final test accuracy `86%`.
  - epoch time after the first epoch was about `1.4-2.0s`; first epoch was `3.85s`.
- Run `runs/20260520_paper_like_100pc_r10_s100_e20`:
  - settings: train `100/class`, test `10/class`, reads `10/10`, sweeps `100/100`, epochs `20`, beta `5`.
  - best test accuracy `90%` at epochs 16-17; final test accuracy `89%`.
  - training accuracy reached `96.9%`; final test loss `1.78`.
  - epoch time after startup was about `8.5-12s`, first epoch `15.6s`.
  - saved best checkpoint: `runs/20260520_paper_like_100pc_r10_s100_e20/best_model.bin`.
  - saved final checkpoint: `runs/20260520_paper_like_100pc_r10_s100_e20/final_model.bin`.
- Larger held-out evaluation of the best checkpoint:
  - run `runs/20260520_paper_like_100pc_r10_s100_e20_eval100pc` loaded the best checkpoint and evaluated `100/class` balanced test images.
  - accuracy was `84.4%`, loss `2.53`, prediction counts `103-98-103-102-91-114-84-82-82-141`.

What made this work:

- Use the same model class as the paper: pixels are continuous inputs that create hidden fields; the sampled Ising machine is only hidden plus output.
- Use 40 output spins, not 10, so each class has four output replicas.
- Use an output-field nudge (`bias_1 - beta * target`) rather than the earlier quadratic clamping term.
- Use the paper's one-sided correlation-difference update.
- Negate graph couplings and fields when installing paper parameters into this package's Hamiltonian, because the sign convention is opposite to `dimod`.

Remaining caveats:

- This is a local experiment file, not yet folded into the threaded manager path.
- It is still stochastic; the 100-example validation accuracy bounces epoch-to-epoch, while the larger 1000-example held-out eval is a better estimate.
- The high skip count late in training is expected under the released code's rule: if the free output already equals the target exactly, the nudged phase contributes zero update for that sample.

Follow-up after first working run:

- Run `runs/20260520_paper_like_200pc_r10_s100_e15` trained on `200/class` and evaluated on `100/class` every epoch.
  - settings: reads `10/10`, sweeps `100/100`, epochs `15`, default paper learning rates.
  - best/final test accuracy `88.6%`, test loss `1.766`.
  - epoch time after the first epoch was about `17.7-21.9s`; first epoch was `28.8s`.
- Run `runs/20260520_paper_like_200pc_r10_s100_e15_lrhalf` repeated the same setup with half learning rates.
  - best/final test accuracy `86.9%`, test loss `2.138`.
  - this did not improve over the paper rates, though late epochs were less violently unstable than early high-rate epochs.
- Run `runs/20260520_paper_like_200pc_best_eval100pc_r30` loaded the best default 200/class checkpoint and evaluated with 30 free reads.
  - accuracy `88.5%`, loss `1.786`, essentially matching the 10-read final result.
  - interpretation: at this point the remaining gap is mostly not free-phase read count; parameters/data/architecture are the next levers.

Current best local result:

- `120` hidden spins, `40` output spins, paper-style field nudge and one-sided update.
- Best checkpoint: `runs/20260520_paper_like_200pc_r10_s100_e15/best_model.bin`.
- Balanced `100/class` test accuracy: `88.6%` with 10 reads, `88.5%` with 30 reads.

Next useful probes if we continue later:

- More training data per class, because the model is now fitting 200/class very well and the larger eval is still below the small validation peak.
- Slightly larger hidden layer only after the paper-size baseline is exhausted; the current result already validates the 120-hidden architecture.
- Confusion diagnostics per digit, because the prediction counts show some residual class bias depending on epoch.

Second follow-up after `Same message` prompt:

- Added per-class accuracy and a full confusion matrix to `mnist_paper_like_ep.jl` CSV rows. This made the later failures interpretable instead of just aggregate accuracy noise.
- Continued the best `200/class` checkpoint on `500/class` training data with half paper learning rates:
  - run: `runs/20260520_paper_like_continue500pc_lrhalf_e8`.
  - settings: train `500/class`, test `100/class`, reads `10/10`, sweeps `100/100`, epochs `8`, loaded `runs/20260520_paper_like_200pc_r10_s100_e15/best_model.bin`.
  - best accuracy `90.0%` at epoch 4; final accuracy `89.9%` at epoch 8.
  - best epoch per-class accuracies: `0.95-0.99-0.90-0.92-0.87-0.83-0.88-0.84-0.91-0.91`.
- Evaluated the half-rate 500/class best checkpoint with 30 free reads:
  - run: `runs/20260520_paper_like_continue500pc_lrhalf_best_eval100pc_r30`.
  - accuracy `89.9%`, loss `1.626`.
  - per-class accuracies: `0.95-0.99-0.90-0.92-0.87-0.82-0.88-0.85-0.90-0.91`.
  - interpretation: the 90% result is robust to more free reads, not a single-evaluation accident.
- Tried a default-rate continuation from the same 200/class checkpoint on `500/class`:
  - run: `runs/20260520_paper_like_continue500pc_defaultlr_e5`.
  - best was only the loaded epoch-0 checkpoint at `88.7%`; training epochs degraded/stayed below that.
  - class balance became unstable, especially for digits 8 and 9 in later epochs.

Current best:

- `runs/20260520_paper_like_continue500pc_lrhalf_e8/best_model.bin`.
- Balanced `100/class` test accuracy: `90.0%` with 10 free reads, `89.9%` with 30 free reads.
- This is now a real paper-style MNIST learning loop, and the diagnostics point to normal residual model/parameter limitations rather than a broken learning implementation.

## 2026-05-22 Manager Fixes and Paper-Style Reproduction

Stock shared manager path:

- Fixed two field-input bugs in the shared `MNISTContrastiveStep` route:
  - `apply_input_pattern!` now disables the input layer, so the precomputed
    input-field path matches `apply_input`.
  - `MNISTContrastiveStep` now adds the missing input-to-hidden bilinear
    gradient terms when input is represented by a field and the input state is
    cleared.
- A direct gradient diagnostic after the fix reported nonzero first-layer
  gradient: `input_edge_grad_norm=6.72`, `other_grad_norm=2.15` on one sample.
- The fixed stock route reduces MSE but still does not classify reliably:
  - `runs/20260522_mnist_field_input_fix_2048x8`: train-probe MSE improved
    from `24.41` to about `21.45`, but accuracy stayed near random.
  - Balanced target, sign, and stronger-weight sweeps did not resolve this.

Sanity checks:

- `learning_sweeps/MNISTRawPixelRidgeCheck.jl` gives `65%` balanced validation
  accuracy for direct raw-pixel ridge on `100/class` train and `30/class`
  validation, so the data/label path is fine.
- `mnist_hidden_ridge_warmstart.jl` now uses the shared field-input path and
  writes hidden-output readout weights symmetrically. Random relaxed hidden
  states were still weak, supporting the conclusion that the stock
  quadratic-clamp graph formulation is the problem, not just labels.

Fresh working reproduction:

- Re-ran `mnist_paper_like_ep.jl` with train `100/class`, test `10/class`,
  reads `3/3`, sweeps `50/50`, epochs `8`.
- Run directory: `runs/20260522_paper_like_repro_100pc_r3_s50_e8`.
- Best balanced test accuracy: `83%` at epoch `6`.
- Re-ran the stronger `200/class`, `10/10` reads, `100/100` sweeps,
  `15` epoch point.
  - Run directory: `runs/20260522_paper_like_repro_200pc_r10_s100_e15`.
  - Best balanced test accuracy: `85.0%` at epoch `8`.
  - First epoch took `29.1s`; later epochs were about `18-20s`.
- This confirms the paper-style route is still the working MNIST baseline:
  input-to-hidden as external fields, hidden/output sampled graph, output-field
  nudge, one-sided correlation update, and sign-negated graph installation.

## 2026-05-22 Local Paper-Style Architecture Sweep

New file:

- `learning_sweeps/MNISTLocalPaperLikeEP.jl`.

Schema:

- Uses the paper-style one-sided EP rule, not the stock quadratic clamp.
- Input pixels are external fields into hidden1.
- The sampled graph is hidden1-hidden2-output.
- Trainable:
  - local input-to-hidden1 map;
  - local hidden1-to-hidden2 couplings;
  - dense hidden2-to-output readout;
  - biases.
- Optional fixed local intra-layer couplings exist for hidden/output layers.

Results:

- `runs/20260522_local_paper_h14_r2_s30_e5` collapsed to mostly one class.
- `runs/20260522_local_paper_h14_r4_s50_lr003_nointernal` reached `32%`
  small-test accuracy. Wider radius, lower LR, and no fixed intra-layer
  couplings were better.
- `runs/20260522_local_paper_h28_h14_r5_s50_lr003_nointernal` reached `62%`
  on a `5/class` test.
- Adding weak fixed intra-layer couplings in
  `runs/20260522_local_paper_h28_h14_r5_s50_lr003_internal005` hurt, topping
  out at `46%`.
- `runs/20260522_local_paper_h28_h28_r5_s50_lr003_nointernal` collapsed more
  than the smaller second hidden layer.
- Scaling the good asymmetric shape:
  - `runs/20260522_local_paper_h28_h14_r5_s50_lr003_50pc`: best `74%` on
    `10/class` test.
  - `runs/20260522_local_paper_h28_h14_r5_s50_lr003_100pc`: best `74.5%` on
    `20/class` test.
  - half learning rate run topped out at `68%`, so `0.003` is currently better.
- Best current local shape:
  - `runs/20260522_local_paper_h28_h11_r5_s50_lr003_100pc`
  - architecture `784 -> 121 -> 40` after the input-field map.
  - best `20/class` test accuracy `75.5%`.
  - loaded best checkpoint and evaluated `100/class` test in
    `runs/20260522_local_paper_h28_h11_best_eval100pc`: accuracy `70.6%`.

Interpretation:

- The local architecture can learn under the paper-style nudge/update.
- Current local results are below the nonlocal `120 -> 40` paper baseline, but
  the best local run is already clearly above chance and has balanced-ish
  prediction counts.
- For this local setup, fixed intra-layer couplings are not automatically good;
  they should stay an explicit sweep variable rather than being assumed helpful.

## 2026-05-20 10x Hidden Manager Batch Profile

Profile command target:

- Script: `Profiling.jl`.
- Run directory: `runs/20260520_profile_10xhidden_15_16_b64_s5`.
- Julia threads: `16`.
- Graph: `784 -> 7840 -> 40` (`hidden=7840`, `output_replicas=4`).
- Batch size: `64`, `MINIT=1`.
- Relaxation: `5` active-layer sweeps, computed as `39400` LocalLangevin steps per phase.
- One warmup batch, three measured batches.

Measured averages:

- 15 workers:
  - manager run: `4.928s` per batch.
  - reset buffers: `0.024s`.
  - collect gradient: `0.034s`.
  - optimiser: `0.115s`.
  - broadcast/sync: `0.032s`.
  - approximate full minibatch time: `5.133s`.
  - manager CPU utilization: about `15.18` CPU threads.
- 16 workers:
  - manager run: `5.065s` per batch.
  - reset buffers: `0.024s`.
  - collect gradient: `0.037s`.
  - optimiser: `0.081s`.
  - broadcast/sync: `0.035s`.
  - approximate full minibatch time: `5.241s`.
  - manager CPU utilization: about `16.49` CPU threads.

Interpretation:

- For the 10x hidden graph, 15 workers is slightly faster than 16 workers in this short profile.
- The batch is dominated by the worker manager run, not data loading, target writing, gradient collection, or parameter sync.
- Parameter sync for this graph is around `32-35ms` for 17-18 synced graphs, so sync is no longer the bottleneck.

## 2026-05-20 10x Hidden 500-Sweep Batch Timing

Profile target:

- Script: `Profiling.jl`.
- Run directory: `runs/20260520_profile_10xhidden_15w_b64_s500`.
- Julia threads: `16`.
- Workers: `15`.
- Graph: `784 -> 7840 -> 40`.
- Batch size: `64`, `MINIT=1`.
- Relaxation: `500` active-layer sweeps, computed as `3,940,000` LocalLangevin steps per phase.
- Warmup batches: `0`; measured one full batch directly.

Timing:

- manager run: `455.890s`.
- reset buffers: `0.059s`.
- collect gradient: `0.258s`.
- optimiser: `0.552s`.
- broadcast/sync: `0.048s`.
- approximate real minibatch time: `456.81s` (`7.61min`) for batch size 64.
- manager CPU utilization: about `15.18` CPU threads.

Other setup/profiling measurements:

- graph build: `3.58s`.
- trainer build: `30.42s`.
- MNIST load: `4.50s`.
- isolated worker state write diagnostic: `1.00s` for the batch.
- isolated input/target apply diagnostic: `0.26s` for the batch.

Interpretation:

- At 500 sweeps, the batch is almost entirely dynamics time. Sync and gradient collection are negligible relative to relaxation.
- The 5-sweep profile scaled roughly as expected: 500 sweeps puts one 64-example batch around 7.6 minutes with 15 workers.

## 2026-05-20 Process Overhead and 32-Thread Scaling

Single direct process vs one manager worker:

- Added script: `ProcessOverhead.jl`.
- Run directory: `runs/20260520_process_vs_manager_worker_s5`.
- Julia threads: `16`.
- Same `MNISTContrastiveStep` worker algorithm in both cases.
- Both cases use one MNIST job, `output_replicas=4`, `5` active-layer sweeps, one warmup, three measured repeats.

Measured averages after warmup:

- Hidden `120`:
  - direct process: `0.00164s`.
  - one-worker manager: `0.00157s`.
  - measured overhead: effectively zero/noise (`-0.00007s`).
- Hidden `7840`:
  - direct process: `0.20180s`.
  - one-worker manager: `0.20644s`.
  - measured overhead: about `0.00464s` (`~2.4%`) for a single job at this short 5-sweep setting.

Interpretation:

- The manager/worker wrapper is not adding meaningful overhead compared with the dynamics. At realistic 500-sweep settings this overhead is completely drowned out.
- The previous batch timings are dominated by the actual Process work, not the `ProcessManager` lifecycle.

31/32 worker scaling with half batch:

- Run directory: `runs/20260520_profile_10xhidden_15_16_31_32_b32_s500`.
- Julia threads: `32`.
- Graph: `784 -> 7840 -> 40`.
- Batch size: `32`, one measured batch, no warmup batch.
- Relaxation: `500` active-layer sweeps = `3,940,000` LocalLangevin steps per phase.

Measured batch times:

- 15 workers:
  - manager run `236.72s`, approximate full minibatch `237.60s`, manager CPU threads `14.81`.
- 16 workers:
  - manager run `230.15s`, approximate full minibatch `230.36s`, manager CPU threads `16.66`.
- 31 workers:
  - manager run `237.87s`, approximate full minibatch `238.16s`, manager CPU threads `25.39`.
- 32 workers:
  - manager run `226.42s`, approximate full minibatch `226.72s`, manager CPU threads `27.28`.

Interpretation:

- Going from 16 to 32 workers on a 32-example batch only improved this one-batch timing by about `1.6%` (`230.36s -> 226.72s`).
- 31 workers was worse than 16 and 32 in this run.
- CPU usage rose to about `25-27` effective CPU threads for 31/32 workers, so more logical cores were used, but the runtime barely improved. This suggests the 10x hidden 500-sweep job is hitting memory bandwidth/cache/scheduling limits rather than simply lacking workers.
- Sync remained tiny (`~0.05s` for 15/16, `~0.08s` for 31/32), so sync is still not the bottleneck.

## 2026-05-20 Long-Relaxation Single Process vs One Worker Latency

Updated `ProcessOverhead.jl` to report both external end-to-end latency and the internal `Process` runtime clock:

- Direct process external latency is measured from just before `run(worker)` through `wait(worker); fetch(worker)`.
- Direct process internal time is `Processes.runtime(worker)` after `wait/fetch`.
- Manager external latency is `run!(manager, (job,))` for a one-worker `ProcessManager`.
- Manager internal time is `Processes.runtime(manager_worker)` after `run!` returns.

Run target:

- Directory: `runs/20260520_process_vs_manager_worker_10x_s500_latency`.
- Julia threads: `16`.
- Graph: `784 -> 7840 -> 40`.
- One MNIST job.
- Relaxation: `500` active-layer sweeps = `3,940,000` LocalLangevin steps per phase.
- One warmup, two measured repeats.

Measured averages after warmup:

- Direct process:
  - external `run + wait + fetch` latency: `16.5391s`.
  - internal `Processes.runtime(worker)`: `16.5391s`.
  - external minus internal: about `0.000057s`.
  - `run(worker)` launch call itself: about `0.0000018s`.
- One-worker manager:
  - external `run!(manager, (job,))` latency: `17.1197s`.
  - internal worker runtime: `17.1197s`.
  - external minus internal: about `0.000045s`.
- Direct-vs-manager latency difference: manager was `0.581s` slower on average (`~3.5%`) in this two-sample run.

Interpretation:

- The process internal clock and external wait/fetch latency match to within microseconds once compiled, so the process clock is a good proxy for actual process run latency.
- The one-worker manager does not add measurable scheduler/wait/fetch overhead outside the worker runtime. The small direct-vs-manager difference appears inside the worker runtime and is likely run-to-run stochastic/dynamics variability plus manager prepare/reset context, not a large lifecycle tax.
- For batch-scale 500-sweep runs, the overhead is still negligible compared with total dynamics time.

## 2026-05-20 Direct Process Concurrency Check

Purpose:

- Decide whether the poor 10x-hidden scaling is caused by `ProcessManager` scheduling/job distribution or by running many dynamics processes concurrently.

New script:

- `DirectProcessConcurrency.jl`.
- It builds the same `MNISTContrastiveStep` workers as the manager path, but does not wrap them in a `ProcessManager`.
- For `N` workers, it writes one job into each worker, calls `run(worker)` on all workers, then `wait/fetch`es them all.

Run target:

- Directory: `runs/20260520_direct_concurrency_10x_16_32_s500`.
- Julia threads: `32`.
- Graph: `784 -> 7840 -> 40`.
- Relaxation: `500` active-layer sweeps = `3,940,000` LocalLangevin steps per phase.

Direct process results:

- 16 direct processes, 16 jobs, one wave:
  - total external latency: `116.62s`.
  - mean internal process runtime: `114.99s`.
  - max internal process runtime: `116.26s`.
- 32 direct processes, 32 jobs, one wave:
  - total external latency: `226.77s`.
  - mean internal process runtime: `222.53s`.
  - max internal process runtime: `226.77s`.

Comparison to ProcessManager batch-32 profile:

- Manager with 16 workers and 32 jobs took `230.15s`. That is two waves; direct 16-process one-wave latency predicts `2 * 116.62s = 233.24s`, matching the manager result.
- Manager with 32 workers and 32 jobs took `226.42s`; direct 32-process one-wave latency was `226.77s`, essentially identical.

Interpretation:

- The poor scaling is not caused by `ProcessManager` scheduling or job distribution. Direct concurrent processes without the manager show the same slowdown.
- Chunking jobs would not help the 32-worker/batch-32 case, because that case is already one scheduling wave. There are no extra waves to eliminate, and task launch overhead is tiny compared with the 226s dynamics latency.
- For the 16-worker/batch-32 case, chunking two jobs per worker into one task could remove one wave boundary and one task launch per worker, but the direct result predicts the same total runtime because the two waves are dominated by dynamics. It may also hurt load balancing if worker runtimes vary.
- The bottleneck is inside concurrent dynamics execution, most likely memory/cache/bandwidth pressure from many large `7840`-hidden graph traversals, not manager overhead.

Attempted a 10x-hidden pointer audit for worker graph arrays, but the command timed out during setup. It was killed to avoid leaving background Julia processes.

## 2026-05-20 Manual Raw Langevin Context Scheduling

Purpose:

- Check whether poor 10x-hidden scaling comes from `ProcessManager` or full `Process` scheduling, by bypassing both and manually spawning tasks that only run `Processes.step!(langevin, context)`.

New script:

- `ManualTests/ManualLangevinContextScheduling.jl`.
- It builds independent graph/context copies, fixes one MNIST input on each graph, warms each context with one step, then schedules one task per context.
- The measured loop is only `Processes.step!(langevin, context)` repeated for `500` active-layer sweeps.
- This uses `Processes.init/step!` directly, but avoids `Process`, `ProcessManager`, `MNISTContrastiveStep`, job writing, gradient accumulation, target application, and sync.

Run target:

- Directory: `runs/20260520_manual_context_10x_16_32_s500`.
- Julia threads: `32`.
- Graph: `784 -> 7840 -> 40`.
- One Langevin phase only.
- Relaxation: `500` active-layer sweeps = `3,940,000` raw `step!` calls per context.

Results:

- 16 contexts:
  - single context baseline: `3.900s`.
  - spawned total latency: `26.312s`.
  - per-task times ranged `25.700s` to `26.279s`.
  - speedup vs serial execution of 16 contexts: `2.37x`.
  - slowdown vs ideal one-wave time: `6.75x`.
- 32 contexts:
  - single context baseline: `3.990s`.
  - spawned total latency: `51.820s`.
  - per-task times ranged `49.137s` to `51.820s`.
  - speedup vs serial execution of 32 contexts: `2.46x`.
  - slowdown vs ideal one-wave time: `12.99x`.

Comparison to full contrastive process timings:

- A full MNIST contrastive job has roughly three relaxation phases, so the manual one-phase 32-context timing predicts about `3 * 51.82s = 155.46s` before reset/capture/gradient costs.
- The previous direct 32-process full contrastive one-wave timing was `226.77s`.
- The manual test is faster because it removes the extra phases' bookkeeping and gradient work, but it still shows the same qualitative scaling failure before `ProcessManager` enters the picture.

Interpretation:

- The bad scaling is already present in raw concurrent `LocalLangevin` stepping on independent contexts.
- That makes a manager scheduling bug unlikely as the primary cause.
- The bottleneck is probably inside the concurrent step workload itself: memory/cache/bandwidth pressure from many large 10x-hidden graphs, random active-spin access, and/or shared runtime resources used by the step implementation.

Follow-up:

- Moved the `@elapsed` timing into a regular helper function and changed the spawned call to use direct `Threads.@spawn` dollar interpolation:
  - `Threads.@spawn timed_spawn_run!(...)`.
- Run directory: `runs/20260520_manual_context_spawn_interp_10x_16_32_s500`.
- 16 contexts:
  - single context baseline: `4.131s`.
  - spawned total latency: `26.555s`.
  - speedup vs serial execution of 16 contexts: `2.49x`.
- 32 contexts:
  - single context baseline: `4.063s`.
  - spawned total latency: `52.107s`.
  - speedup vs serial execution of 32 contexts: `2.50x`.

Dollar interpolation does not materially change the result. The earlier closure capture was not the source of the poor scaling.

## 2026-05-21 MNIST 10x Crash/Memory Diagnostic

Purpose:

- Check whether the intermittent 10x-hidden worker crash is caused by memory exhaustion, invalid sparse/active indices, or the LoopVectorization sparse contraction.

New script:

- `MNISTKernelDiagnostics.jl` validates CSC row/column invariants, active index ranges, and optionally runs single-thread `LocalLangevin`.
- `MNISTProcessReuseDebug.jl` runs multiple jobs through reusable MNIST `Process` workers and can optionally replace `column_contraction` with a checked non-`@turbo` method for one Julia session.

Results:

- Static CSC checks on `784 -> 7840 -> 40` passed:
  - states: `8664`.
  - nnz: `12920320`.
  - row range: `1:8664`.
  - active range: `1:8664` before input clamping.
- Checked contractions over all active columns succeeded.
- Single-thread `LocalLangevin` on the 10x graph ran successfully.
- Eight independent 10x graphs and direct concurrent `LocalLangevin`/`MNISTContrastiveStep` calls ran successfully.
- Temporarily replacing `@turbo` in `src/Utils/LoopVectorization.jl` with normal Julia loops allowed an 8-worker, two-jobs-per-worker reuse diagnostic to complete:
  - memory after workers: about `3292 MiB` working set, `5626 MiB` private.
  - elapsed: `3.57s` for 5 active-layer sweeps.
- Restoring `@turbo` and rerunning the same 8-worker diagnostic also completed:
  - elapsed: `3.28s`.
- The current manager path with `MakeEachWorker`, 10x hidden, 8 workers, batch 16, 5 sweeps completed:
  - run time: `2.28s`.
- The same manager path with 16 workers, batch 32, 5 sweeps completed:
  - run time: `3.65s`.
- A 32-worker one-job-per-worker memory smoke completed:
  - memory after workers: about `6977 MiB` working set, `9708 MiB` private.
  - memory after run: about `6960 MiB` working set, `9914 MiB` private.

Interpretation:

- This does not look like immediate out-of-memory on the tested 10x setup; 32 workers fit in about `10 GiB` private memory for a short run.
- The fixed sparse matrix row indices and cached active spin indices are in range in the checked cases.
- `@turbo` remains a plausible crash amplifier because it can turn any intermittent stale/bad index into an access violation, but the failure did not reproduce after the latest Processes refactor and current worker creation path.
- The active cache changes from `8664` before the first job to `7880` after input clamping, which is expected for MNIST input-as-fixed-state: `8664 - 784 = 7880`.

## 2026-05-21 OC Warm Stability Retest

Purpose:

- Recheck the previously failing 32-worker, 10x-hidden manager path after reboot/voltage changes and after warming the machine.
- Validate every sparse row index before any worker dynamics starts.

Setup:

- Architecture: `784 -> 7840 -> 40`.
- Workers: `32`.
- Batch size: `32`.
- Relaxation: `25` layer sweeps.
- Worker creation: `MakeEachWorker`.
- Scheduler mode: `greedy`.

Results:

- Construction-only validation of 64 independently built graphs passed before this retest:
  - states: `8664`.
  - nnz: `12920320`.
  - row range: `1:8664`.
- Pre-run manager topology validation passed for:
  - the prototype graph.
  - all 32 worker graphs.
  - the validation graph.
- A single manager batch completed in `12.46s`.
- A three-batch repeat completed without row-index failure:
  - batch 1: `12.33s`.
  - batch 2: `11.77s`.
  - batch 3: `11.84s`.
- With Doom running in the background to add memory/system load, the same three-batch run also completed without row-index failure:
  - batch 1: `15.48s`.
  - batch 2: `14.98s`.
  - batch 3: `15.58s`.
- Repeating the same Doom-background run reproduced the failure after clean pre-run validation:
  - batch 1 completed in `15.65s`.
  - batch 2 completed in `14.57s`.
  - the process crashed during batch 3.
  - stack entered `LoopVectorization.@turbo` in `column_contraction` from `weighted_neighbors_sum -> Bilinear.calculate -> LocalLangevin.step!`.
  - the Julia process exited with `EXCEPTION_ACCESS_VIOLATION`.
- After commenting out `@turbo` in `column_contraction` and tweaking the OC, the same 32-worker three-batch run completed:
  - batch 1: `17.27s`.
  - batch 2: `16.40s`.
  - batch 3: `16.59s`.
  - pre-run topology validation again passed for prototype, all 32 workers, and validation graph.
- A longer no-`@turbo` run with the same settings completed 10 consecutive batches:
  - batch times ranged from `15.93s` to `17.24s`.
  - no bounds error or access violation occurred.

Interpretation:

- The manager is not starting from an already-invalid sparse matrix in this retest.
- The failure can appear after clean graph construction and after multiple successful batches.
- That makes deterministic graph construction an unlikely cause.
- The fault is either an intermittent out-of-bounds index during dynamics that `@turbo` turns into an access violation, or a real memory/hardware corruption under this workload.

## 2026-05-21 MNIST Manager Scaling Pass

Purpose:

- Restore `@turbo` after the no-`@turbo` stability pass.
- Compare manager scheduling modes, old spawn scheduling, and raw context scheduling.
- Check whether the bad scaling is caused by poor job distribution, manager overhead, or the MNIST graph workload itself.

`120` hidden, batch `256`, `1000` sweeps:

- `runthreaded!` modes were essentially tied:
  - 16 workers: `normal 3.48s`, `dynamic 3.47s`, `greedy 3.36s`.
  - 32 workers: `normal 3.13s`, `dynamic 3.09s`, `greedy 3.04s`.
- Old spawn path was essentially the same:
  - 16 workers: `3.48s`.
  - 32 workers: `3.14s`.

Interpretation:

- New manager scheduling, old spawn scheduling, dynamic scheduling, and greedy scheduling all land in the same range.
- That makes manager scheduling overhead or job dispatch style unlikely to be the main limiter for this workload.

Raw manager/thread benchmark, nearest-neighbor `32x32`, `1000` sweeps, 32 contexts:

- Plain `Threads.@spawn`: `0.166s`.
- `ProcessManager` normal: `0.184s`.
- `runthreaded Dynamic`: `0.166s`.
- `runthreaded Static`: `0.146s`.
- `runthreaded Greedy`: `0.151s`.
- Speedup vs serial ranged from about `12.5x` to `15.8x`.

Interpretation:

- The generic manager/threading machinery can scale well on this machine.
- The poor MNIST scaling is not a general `ProcessManager` failure.

Manual raw MNIST-context stepping, `120` hidden, `1000` sweeps:

- 16 contexts:
  - one context: `0.087s`.
  - spawned 16 contexts: `0.160s`.
  - speedup vs serial contexts: `8.74x`.
- 32 contexts:
  - one context: `0.088s`.
  - spawned 32 contexts: `0.248s`.
  - speedup vs serial contexts: `11.35x`.

Interpretation:

- Raw MNIST Langevin stepping scales better than the full contrastive manager batch, but it still flattens well before ideal 32x.
- The remaining difference is likely the full contrastive job structure: three relaxation phases, reset/input/target writes, state captures, and gradient accumulation.

Worker sweep, `120` hidden, batch `64`, `1000` sweeps, dynamic scheduling:

- 1 worker: `11.43s`.
- 2 workers: `5.36s`.
- 4 workers: `2.87s`.
- 8 workers: `1.59s`.
- 16 workers: `0.96s`.
- 32 workers: `0.85s`.

Interpretation:

- Scaling is good up to 16 workers, then mostly flat.
- This looks like a physical-core / memory-bandwidth knee, not poor job spreading.

Knee check, `120` hidden, batch `96`, `1000` sweeps, dynamic scheduling:

- 12 workers: `2.76s`.
- 15 workers: `1.64s`.
- 16 workers: `1.50s`.
- 24 workers: `1.20s`.
- 31 workers: `1.32s`.
- 32 workers: `1.20s`.

Interpretation:

- The measurements are somewhat noisy, but using more than 16 workers can still help a little on small `120`-hidden graphs.
- There is no sign that 31 workers is materially better than 32.

Larger x1 hidden test, `784` hidden, batch `64`, `250` sweeps, dynamic scheduling:

- 1 worker: `27.92s`.
- 8 workers: `8.59s`.
- 16 workers: `6.46s`.
- 32 workers: `10.32s`.
- Repeating the 16-worker case with Julia started as `-t 16` gave `7.12s`, slightly slower than 16 workers under `-t 32`.

Interpretation:

- Larger MNIST connectivity makes 32 workers worse than 16.
- That points to cache/memory pressure from sparse CSC column contraction over dense bipartite blocks, not scheduling imbalance.

Likely fixes to try next:

- Prefer about 16 workers for the larger MNIST graphs; 32 workers only helps the small `120` graph slightly and can hurt the `784` graph.
- Keep the `120` hidden architecture for learning runs unless there is a specific reason to stress-test larger hidden layers.
- Add a specialized dense/layered field-cache path for layered MNIST graphs:
  - maintain `h = J * s + b` or per-layer field blocks inside the dynamics context.
  - after one spin update, update affected neighbor fields by `Δs * J[:, spin]`.
  - calculate local derivatives from cached fields instead of scanning CSC columns every step.
- For all-to-all input-hidden and hidden-output layers, consider storing block weights as dense matrices in the learning context, while preserving the graph API for display and parameter sync.
- Chunking multiple samples into one manager job is unlikely to fix the scaling bottleneck; the current dynamic scheduler already keeps all workers active and old spawn vs runthreaded gives the same timings.

## 2026-05-21 MNIST Field Cache Probe

New file:

- `ManualTests/MNISTFieldCacheProbe.jl`.

Purpose:

- Test whether the current local derivative bottleneck is the repeated sparse
  column scan.
- Compare two cost models:
  - `scan`: two sparse column scans per local accepted step, matching the
    current pre/post derivative shape.
  - `cache`: one cached field lookup plus one neighbor field update after the
    spin change.

Short probe, `200` sweeps:

- `120` hidden:
  - 16 contexts:
    - scan single: `0.025s`, parallel: `0.044s`.
    - cache single: `0.0085s`, parallel: `0.022s`.
  - 32 contexts:
    - scan single: `0.026s`, parallel: `0.041s`.
    - cache single: `0.0084s`, parallel: `0.032s`.
- `784` hidden:
  - 16 contexts:
    - scan single: `0.166s`, parallel: `0.248s`.
    - cache single: `0.056s`, parallel: `0.133s`.
  - 32 contexts:
    - scan single: `0.170s`, parallel: `0.693s`.
    - cache single: `0.058s`, parallel: `0.643s`.

Longer probe, `784` hidden, `1000` sweeps:

- 16 contexts:
  - scan single: `0.836s`, parallel: `1.035s`.
  - cache single: `0.316s`, parallel: `0.790s`.
- 32 contexts:
  - scan single: `0.838s`, parallel: `3.487s`.
  - cache single: `0.309s`, parallel: `3.079s`.

Interpretation:

- A field cache is clearly useful inside one context:
  - about `3x` faster for both `120` and `784` hidden.
- At 16 contexts, the cache still helps:
  - `784` hidden long probe improves from `1.035s` to `0.790s`.
- At 32 contexts, both scan and cache are memory-bound on the larger graph:
  - cache is only modestly faster than scan.
  - cache scaling is worse because it changes the workload from random reads to
    many random writes into each context's field vector.

What this suggests:

- The current sparse-column derivative path is expensive, but a naive per-spin
  sparse field cache is not enough.
- The better fix is a layer-block cache:
  - Precompute the fixed input contribution to hidden fields once per sample:
    `h_hidden_from_input = W_input_hidden' * x`.
  - During relaxation, keep only hidden/output dynamic fields live.
  - When a hidden spin changes, update output fields with a contiguous
    hidden-output weight slice.
  - When an output spin changes, update hidden fields with a contiguous
    hidden-output weight slice.
  - Avoid repeatedly scanning the 784 fixed input connections for every hidden
    local derivative.
- For the paper-like `120` hidden architecture, this should be very cheap:
  - dynamic layer size is only `120 + 40`.
  - fixed input can be folded into per-sample hidden biases.
- For larger hidden layers, a dense block storage layout should be better than
  CSC because updates become contiguous `axpy`-style operations instead of
  random row-indexed sparse writes.

Concrete next implementation direction:

- Add a local experiment-only `LoopAlgorithm` for MNIST with explicit dense
  layer blocks:
  - `W_xh`, `W_ho`, `b_h`, `b_o`.
  - cached fields `field_h`, `field_o`.
  - state arrays `x`, `h`, `o`.
  - one LocalLangevin-like update over hidden/output only.
- Use the existing graph only for parameter storage/display and synchronize the
  dense blocks at minibatch boundaries.
- If that performs well, move the block-cache dynamics into source as a proper
  specialized algorithm rather than complicating the generic graph dynamics.

## 2026-05-21 Block-Cache Scaling Follow-Up

User target:

- Keep trying if the scaling is not close to `14-16x`.

New file:

- `ManualTests/MNISTBlockCacheProbe.jl`.

What changed relative to the sparse field-cache probe:

- Extract dense `Wxh` and `Who` blocks from the MNIST graph.
- Fold the fixed input contribution into hidden fields once:
  - `field_h = b_h + Wxh' * x + Who * o`.
- Maintain only hidden/output cached fields.
- On a hidden update, update output fields from the hidden-output block.
- On an output update, update hidden fields from the hidden-output block.
- Added a second transposed `Woh = Who'` copy so both update directions can use
  contiguous memory.
- Added `@turbo` to the short contiguous field-update loops.
- Compared order modes:
  - `random`: sample one active unit with RNG at every local step.
  - `shuffle`: shuffle active units once per sweep.
  - `cyclic`: deterministic active-unit cycle, best-case scheduling and memory
    locality probe.
- Compared spawned tasks vs `Threads.@threads :static`; static was slightly
  cleaner.

Important results, `784` hidden, 100000 sweeps, static scheduling:

- dual layout, cyclic, no turbo updates:
  - 16 contexts: `11.67x`.
  - 32 contexts: `10.97x`.
- dual layout, cyclic, turbo field updates:
  - 16 contexts: `11.44x`.
  - 32 contexts: `13.48x`.
- same optimized path, Julia `-t 16`, 16 contexts:
  - `12.16x`.
- repeat with 31/32 contexts:
  - 31 contexts: `13.68x`.
  - 32 contexts: `13.70x`.
- worker sweep, optimized cyclic path:
  - 8 contexts: `7.23x`.
  - 16 contexts: `12.42x`.
  - 24 contexts: `12.44x`.
  - 32 contexts: `13.35x`.
- random order, optimized path:
  - 16 contexts: `9.63x`.
  - 32 contexts: `9.58x`.
- shuffled-sweep order, optimized path:
  - 16 contexts: `9.60x`.
  - 32 contexts: `12.28x`.

Important results, `120` hidden, 200000 sweeps, optimized cyclic path:

- 16 contexts: `10.03x`.
- 32 contexts: `12.89x`.

Interpretation:

- The dense block-cache dynamics can get close to the target:
  - best observed `32`-context speedup was about `13.7x`.
- That is a large improvement over the full CSC/graph manager path, where
  `784` hidden with `32` workers was worse than `16`.
- The remaining gap to `14-16x` is mostly from all-core slowdown:
  - in long static runs, total wall time is close to the slowest task time, so
    work is spread well.
  - each task gets slower under full load compared with the isolated single
    context baseline, likely due to all-core clocks and memory/cache pressure.
- Per-step random sampling is too expensive for this optimized loop.
- Shuffling once per sweep is still costly enough to lose much of the gain.
- A production version should probably use either:
  - deterministic/cyclic sweeps, or
  - a cheaper low-overhead pseudo-random sweep order, not per-step
    `MersenneTwister` sampling.

What would plausibly fix the rest:

- Implement the block-cache path as a real MNIST-specific `LoopAlgorithm`.
- Store `Who` and `Woh` in the context to avoid strided row access.
- Use static scheduling for one job per worker in block-cache experiments.
- Avoid random sampling per local step; use cyclic, pre-shuffled, or a cheap
  deterministic permuted order.
- Use the graph as the canonical parameter/display object, but run dynamics on
  dense block arrays and sync at minibatch boundaries.
- For 120-hidden learning, this should be much faster than the current graph
  path even if raw scaling caps around `13x`, because the single-context work is
  drastically smaller.

## 2026-05-21 Sparse-General Cache Follow-Up

Constraint:

- Keep the optimization compatible with general sparse/recurrent Ising graphs.
- Do not rely on MNIST-specific dense layer factorization as the only fix.

Updated file:

- `ManualTests/MNISTFieldCacheProbe.jl`.

Changes:

- Added order modes to the sparse probe:
  - `random`.
  - `shuffle`.
  - `cyclic`.
- Added scheduling modes:
  - `spawn`.
  - `static` via `Threads.@threads :static`.
- Added a topology-general outgoing sparse cache:
  - stores, for each changed spin, the list of cached fields affected by that
    spin and the corresponding weights.
  - this is a CSR-like reverse traversal built from the same sparse matrix.
  - it supports arbitrary sparse/recurrent topology.

Sparse `784` hidden, cyclic/static, `100000` sweeps:

- 16 workers:
  - scan: `13.57x`.
  - naive cache: `7.39x`.
- 32 workers:
  - scan: `11.71x`.
  - naive cache: `4.41x`.

Sparse `784` hidden, cyclic/static, `20000` sweeps, explicit outgoing cache:

- 16 workers:
  - scan: `13.85x`.
  - naive cache: `6.58x`.
  - outgoing cache: `7.51x`.
- 32 workers:
  - scan: `11.22x`.
  - naive cache: `4.63x`.
  - outgoing cache: `4.42x`.

Interpretation:

- For general sparse graphs, the cached-field update is not automatically a
  scaling win.
- The sparse scan path does more arithmetic and is slower for one worker, but
  it is mostly read-only and scales much better.
- The cached sparse path does scattered writes into each context's field vector.
  Under many concurrent contexts this becomes memory/cache-coherence heavy.
- Building an explicit outgoing sparse cache helps at 16 workers but does not
  fix 32-worker scaling.

Sparse-compatible options still worth trying:

- Keep sparse matrices, but add a read-only precomputed static-input field:
  - for fixed/clamped nodes, compute their field contribution once per sample.
  - during relaxation, scan only dynamic-to-dynamic connections.
  - this keeps sparse generality and avoids repeatedly scanning fixed inputs.
- Split sparse adjacency into two sparse pieces per worker context:
  - fixed-source contribution: used once after input/clamp changes.
  - dynamic-source contribution: used inside local dynamics.
  - this supports recurrent dynamic connections.
- Use sparse scan plus better active order:
  - cyclic or cheap shuffled sweeps scale much better than per-step random
    sampling in the probes.
- Avoid sparse cached field updates unless the graph is low-degree enough that
  scattered writes are cheap.
- For dense/high-degree recurrent graphs, a general block/tile sparse format may
  be needed:
  - still sparse and topology-general,
  - but stores neighbors in cache-friendly chunks rather than CSC columns only.

Practical recommendation for now:

- For the current generic sparse backend, the safest improvement is a
  fixed-input/static-node contribution cache, not a full dynamic field cache.
- That should help MNIST because input pixels are fixed during relaxation, while
  still allowing recurrent hidden/output dynamics in later experiments.

## 2026-05-21 Contrastive Phase Timing

Question:

- Why does the real MNIST batch not look as good as the synthetic sparse timing?

New file:

- `ManualTests/MNISTContrastivePhaseTiming.jl`.

What it measures:

- The actual `MNISTContrastiveStep` body, without `ProcessManager` scheduling.
- Timed phases are reset/input, free relaxation, state capture, plus setup,
  plus relaxation, minus setup, minus relaxation, and gradient accumulation.

`784` hidden, batch `64`, `250` sweeps, Julia `-t 32`, default
`LocalLangevin(order = :random)`:

- 1 worker:
  - total batch: `27.72s`.
  - relaxation: `27.46s`.
  - gradient accumulation: `0.185s`.
- 16 workers:
  - total batch: `7.05s`.
  - relaxation sum over workers: `104.25s`.
  - max worker relaxation: `8.17s`.
  - gradient accumulation sum: `0.325s`.
- 32 workers:
  - total batch: `10.95s`.
  - relaxation sum over workers: `339.19s`.
  - max worker relaxation: `12.19s`.
  - gradient accumulation sum: `0.588s`.

Same setup with `LocalLangevin(order = :cyclic)`:

- 1 worker:
  - total batch: `25.75s`.
  - relaxation: `25.49s`.
  - gradient accumulation: `0.187s`.
- 16 workers:
  - total batch: `5.05s`.
  - relaxation sum over workers: `76.42s`.
  - max worker relaxation: `5.57s`.
  - gradient accumulation sum: `0.316s`.
- 32 workers:
  - total batch: `8.82s`.
  - relaxation sum over workers: `273.12s`.
  - max worker relaxation: `10.19s`.
  - gradient accumulation sum: `0.602s`.

Interpretation:

- Gradient accumulation is not the bottleneck for this benchmark.
- Setup/capture costs are also tiny.
- The slowdown is inside relaxation.
- Switching from random active order to cyclic active order helps, but does not
  fix the 32-worker collapse.
- The real `LocalLangevin` path is heavier than the synthetic probe:
  - it runs three relaxation phases per sample: free, plus, and minus;
  - it uses the generic Hamiltonian derivative, not only the bilinear sparse
    matrix term;
  - it recomputes all active derivatives at the start of each sweep;
  - it checks layer bounds and proposal bookkeeping every local update;
  - it reads clamping/magnetic-field data even when those terms are small.
- The synthetic sparse scan measured a tight read-only derivative loop. The
  real workload is that loop plus the full generic dynamics machinery.
- The main scaling issue is still concurrent memory/cache pressure from many
  independent workers repeatedly reading large sparse MNIST coupling matrices.
  It is not manager launch overhead and it is not buffer sync.

Current practical conclusion:

- For the larger `784` hidden graph, use about `16` workers rather than `32`.
- Use `LocalLangevin(order = :cyclic)` or a cheap pre-shuffled order for MNIST
  timing/learning runs unless random order is specifically needed.
- The next sparse-compatible optimization to try is precomputing the fixed
  input contribution once per sample, then relaxing only over dynamic-to-dynamic
  sparse connections.

## 2026-05-21 Staged MNIST Loop Scaling

Question:

- Add one piece of the real worker loop at a time and see which piece kills
  scaling.

New file:

- `ManualTests/MNISTStagedLoopScaling.jl`.

Stages:

- `manual_field_update`:
  - one generic Hamiltonian derivative per local update;
  - deterministic local state update;
  - no post-update derivative, no noise, no sweep-start derivative refresh.
- `manual_prepost_update`:
  - adds post-update derivative calculation.
- `manual_prepost_noise`:
  - adds Langevin noise.
- `manual_prepost_noise_refresh`:
  - adds the sweep-start derivative refresh over all active spins.
- `one_relax`:
  - uses the real `LocalLangevin` step for one relaxation phase.
- `reset_input_one_relax`:
  - adds reset and input application before the one relaxation phase.
- `three_relax`:
  - adds free/plus/minus relaxation phases but no captures/clamping.
- `three_relax_capture`:
  - adds equilibrium/plus/minus state copies.
- `three_relax_clamp`:
  - adds target application and positive/negative clamping.
- `full`:
  - adds contrastive gradient accumulation.

Important setup detail:

- The contexts now fix the input layer before `LocalLangevin` initialization so
  even `one_relax` uses the same hidden/output active set as the real MNIST
  worker.

Inner relaxation ladder, `784` hidden, batch `64`, `100` sweeps, cyclic order:

- `manual_field_update`:
  - 1 worker `1.17s`, 16 workers `0.19s` (`6.18x`), 32 workers `0.35s` (`3.38x`).
- `manual_prepost_update`:
  - 1 worker `2.36s`, 16 workers `0.27s` (`8.92x`), 32 workers `0.38s` (`6.22x`).
- `manual_prepost_noise`:
  - 1 worker `2.40s`, 16 workers `0.28s` (`8.64x`), 32 workers `0.45s` (`5.29x`).
- `manual_prepost_noise_refresh`:
  - 1 worker `3.54s`, 16 workers `0.45s` (`7.97x`), 32 workers `0.71s` (`4.97x`).
- `one_relax`:
  - 1 worker `3.48s`, 16 workers `0.45s` (`7.83x`), 32 workers `0.72s` (`4.86x`).

Longer inner check, `784` hidden, batch `64`, `500` sweeps, cyclic order:

- `manual_prepost_noise_refresh`:
  - 1 worker `17.11s`, 16 workers `2.15s` (`7.97x`), 32 workers `3.62s` (`4.73x`).
- `one_relax`:
  - 1 worker `17.26s`, 16 workers `2.24s` (`7.70x`), 32 workers `3.51s` (`4.92x`).

Outer contrastive ladder, `784` hidden, batch `64`, `100` sweeps, cyclic order:

- `one_relax`:
  - 1 worker `3.46s`, 16 workers `0.46s` (`7.51x`), 32 workers `0.74s` (`4.65x`).
- `reset_input_one_relax`:
  - 1 worker `3.47s`, 16 workers `0.45s` (`7.65x`), 32 workers `0.82s` (`4.20x`).
- `three_relax`:
  - 1 worker `10.41s`, 16 workers `1.37s` (`7.62x`), 32 workers `2.21s` (`4.71x`).
- `three_relax_capture`:
  - 1 worker `10.31s`, 16 workers `1.44s` (`7.18x`), 32 workers `2.19s` (`4.70x`).
- `three_relax_clamp`:
  - 1 worker `10.38s`, 16 workers `1.37s` (`7.59x`), 32 workers `2.05s` (`5.06x`).
- `full`:
  - 1 worker `10.53s`, 16 workers `1.40s` (`7.54x`), 32 workers `2.49s` (`4.23x`).

Interpretation:

- The 32-worker problem is already present in the simplest generic sparse
  derivative stage.
- Adding post-derivative calculation, Langevin noise, sweep-start refresh,
  state copies, clamping, and gradient accumulation changes absolute time but
  does not introduce a new scaling cliff.
- The closest manual approximation to `LocalLangevin`
  (`manual_prepost_noise_refresh`) has almost the same timing as `one_relax`,
  especially at 500 sweeps.
- So the scaling loss is not caused by the outer contrastive-learning code and
  not by a hidden manager sync.
- It is caused by many workers simultaneously reading the large sparse MNIST
  graph through the generic Hamiltonian derivative path.

Practical conclusion:

- For the current sparse graph backend, 16 workers is the useful point for this
  larger MNIST graph.
- 32 workers is consistently slower than 16 workers even when the work is a
  single hand-written relaxation phase.
- The next useful optimization should target sparse derivative memory access,
  not `ProcessManager` scheduling or gradient accumulation.

## 2026-05-21 Worker-Local Arithmetic Process Scaling

Question:

- If a `ProcessAlgorithm` only mutates worker-local state with useless
  arithmetic, does the manager/CPU scale better than the MNIST sparse derivative
  workload?

New file:

- `ManualTests/ArithmeticProcessScaling.jl`.

Algorithm:

- `ArithmeticStateStep` owns one private `Vector{Float64}` per worker.
- Each `step!` loops over that vector for a configurable number of rounds.
- `inner_ops` controls arithmetic per vector element:
  - `inner_ops = 1`: low arithmetic intensity, mostly private vector reads/writes.
  - larger values: more local arithmetic per loaded element.

Manager setup:

- Uses `ProcessManager` and `runthreaded!(..., Dynamic())`.
- One job per worker.
- No shared graph and no shared buffers.

Small private vector, `len = 4096`, `rounds = 5000`:

- `inner_ops = 1`:
  - 1 worker `0.013s`, 16 workers `0.161s` (`1.33x`), 32 workers `0.181s` (`2.37x`).
  - Too short and too little work per element; manager overhead/timing noise dominates.
- `inner_ops = 16`:
  - 1 worker `0.465s`, 16 workers `0.708s` (`10.51x`), 32 workers `0.936s` (`15.90x`).
- `inner_ops = 64`:
  - 1 worker `3.500s`, 16 workers `4.088s` (`13.70x`), 32 workers `4.552s` (`24.61x`).

Large private vector, `len = 1_048_576`, `rounds = 200`, `inner_ops = 1`:

- 1 worker `0.134s`, 16 workers `0.697s` (`3.09x`), 32 workers `2.649s` (`1.62x`).

Interpretation:

- The manager can scale a compute-heavy worker-local `ProcessAlgorithm` well.
  With enough arithmetic per private vector load, 32 workers gives much better
  throughput than 16.
- Private contiguous memory traffic behaves differently:
  - large private arrays with low arithmetic intensity already collapse at 32
    workers.
- MNIST looks more like the low-arithmetic memory case than the compute-heavy
  case, except worse because its sparse derivative reads are irregular and touch
  row indices, weights, and state entries.
- This supports the memory/cache-pressure explanation:
  - pure compute scales;
  - private memory traffic scales poorly at 32;
  - MNIST sparse derivative traffic also scales poorly at 32.

## 2026-05-21 Shared Adjacency Worker Diagnostic

Question:

- Is 32-worker MNIST slow because every worker has its own copied adjacency
  arrays, or because the sparse access pattern is bad regardless?

Source change:

- `MNISTArchitecture` now exposes the existing public `IsingGraph(...; adj=...)`
  constructor entrypoint as an optional `adj` keyword.
- This lets experiments allocate fresh graph state, bias, and clamping buffers
  while sharing the same adjacency object.

Experiment change:

- `ManualTests/MNISTStagedLoopScaling.jl` gained
  `ISING_MNIST_STAGED_ADJ_MODE=copied|shared`.
- `shared` constructs each worker graph through `MNISTArchitecture(...; adj =
  shared_adj)`, so the worker `Bilinear` term points at the same adjacency as
  the graph.

Setup:

- `784` hidden, batch `64`, `500` sweeps, cyclic order.
- Stages:
  - `manual_prepost_noise_refresh`.
  - `one_relax`.

Copied adjacency:

- `manual_prepost_noise_refresh`:
  - 1 worker `16.98s`, 16 workers `2.31s` (`7.34x`), 32 workers `4.44s` (`3.83x`).
- `one_relax`:
  - 1 worker `17.47s`, 16 workers `2.50s` (`7.00x`), 32 workers `3.62s` (`4.83x`).

Shared adjacency:

- `manual_prepost_noise_refresh`:
  - 1 worker `17.29s`, 16 workers `1.29s` (`13.40x`), 32 workers `1.24s` (`13.95x`).
- `one_relax`:
  - 1 worker `17.01s`, 16 workers `1.42s` (`11.97x`), 32 workers `1.25s` (`13.62x`).

Interpretation:

- This is a large effect.
- The bad 32-worker scaling is mostly not caused by the arithmetic in the
  derivative. It comes from every worker repeatedly walking a different copy of
  a large adjacency matrix.
- Sharing the read-only adjacency lets all workers reuse the same matrix data in
  cache much more effectively.
- This is safe for relaxation if workers only read model weights and write their
  own state/buffers.
- It is not safe if worker-local code mutates adjacency weights during the
  minibatch. For the current MNIST contrastive worker, gradients go into worker
  buffers and weights are synchronized after the batch, so shared read-only
  adjacency is a plausible real optimization.

## 2026-05-22 Shared Static Model Data in the Real MNIST Trainer

Question:

- Can the actual `ProcessManager` MNIST trainer share the source graph's
  read-only model arrays, so parameter updates only write the source graph?

Source changes:

- `MNISTArchitecture` accepts:
  - `adj`: shared adjacency.
  - `b`: shared learnable base bias. Arrays are passed with `Force(...)` so the
    term-instantiation path does not copy them.
  - `input_b`: optional second `MagField` used as a worker-local input-pattern
    field.
- `apply_input` now detects a second `MagField`. In that mode the image is
  encoded as the equivalent field generated by fixed input spins:
  `b_j += sum_i J[j, i] * x_i`.
- The custom MNIST worker precomputes that input field once per job and reuses
  it across free, plus, and minus phases.
- `init_mnist_trainer(...; share_static_model_data = true, input_mode = :field)`
  uses the public `ProcessManager` `MakeEachWorker()` mode so every worker is
  constructed directly against the source graph arrays instead of copying a
  template worker.

Smoke check:

- All training workers share `adj(worker_graph) === adj(source_graph)`.
- All training workers share the first/base `MagField.b` with the source graph.
- Each worker still owns its own second/input `MagField.b`.
- One minibatch with shared static model data completed successfully.

Actual trainer batch timing:

- Setup: hidden `784`, output replicas `4`, batch `64`, `500` sweeps, cyclic
  `LocalLangevin`, one warmup batch plus one measured batch.

Copied worker graphs, state input:

- 16 workers: `15.45s`.
- 32 workers: `16.61s`.

Shared adjacency + shared base bias + worker-local input field:

- 16 workers: `6.54s`.
- 32 workers: `4.11s`.

Missing 1-worker baseline for the same `784 -> 784 -> 40`, batch `64`, `500`
sweeps setup:

- Copied worker graph, state input: `51.79s` per batch.
- Shared adjacency + shared base bias + worker-local input field: `50.54s` per
  batch.

End-to-end batch speedup from 1 worker:

- Copied worker graphs:
  - 16 workers: `3.35x`.
  - 32 workers: `3.12x`.
- Shared static model data:
  - 16 workers: `7.73x`.
  - 32 workers: `12.30x`.

`784 -> 120 -> 40`, shared static model data, `500` sweeps:

- Batch `64`, 32 workers:
  - measured minibatch mean: `0.263s`.
  - full no-validation epoch: `242.26s`, about `4.04 min`.
- Batch `256`, 32 workers:
  - measured minibatch mean: `0.947s`.
  - full no-validation epoch: `224.44s`, about `3.74 min`.
- Batch `256`, 1 worker:
  - measured minibatch mean: `12.92s`.
  - extrapolated epoch: about `50.62 min`.

Interpretation:

- This is the first real trainer timing that shows the shared-memory worker
  layout helping the full learning path, not just the staged relaxation loop.
- The worker graphs no longer need parameter writes after a minibatch when
  model arrays are shared. `_broadcast_params!` still updates the source graph
  from optimizer-owned arrays, and worker sync skips arrays that are identical
  to the source arrays.
- The remaining per-job writes are worker-local state, target/clamping storage,
  gradient buffers, and the second input field.

## 2026-05-22 Local Paper-Style Same-Layer Coupling Runs

Question:

- Can the local two-hidden-layer paper-style MNIST experiment benefit from
  learned same-layer connections, instead of fixed random intra-layer
  couplings?

File:

- `learning_sweeps/MNISTLocalPaperLikeEP.jl`.

Source-level experiment change:

- Added optional trainable same-layer couplings:
  - hidden1-hidden1: `weights_11`;
  - hidden2-hidden2: `weights_22`;
  - output-output: `weights_oo`.
- Enable with `ISING_MNIST_LOCAL_PAPER_TRAIN_INTERNAL=true`.
- When enabled, fixed random intra-layer couplings are not installed. The
  same-layer matrices are updated with the same one-sided paper-style EP
  correlation difference as the inter-layer couplings, then masked,
  symmetrized, diagonal-cleared, and clipped.
- Checkpoints now serialize these same-layer matrices. Old checkpoints still
  load; missing same-layer matrices are left at the new model's initialized
  values.

Smoke:

- `runs/20260522_local_paper_traininternal_smoke`
- `8x8 -> 6x6 -> 40`, `1/class`, reads `1/1`, sweeps `2/2`.
- Completed and wrote checkpoints.

Useful local shape:

- `runs/20260522_local_paper_h28_h11_traininternal_100pc`
- Architecture: `28x28 -> 11x11 -> 40`.
- Settings: local radius `5`, same-layer radius `1`, `100/class` train,
  `20/class` test, reads `3/3`, sweeps `50/50`, epochs `12`,
  `W0/W12/W2O lr = 0.003`, `W11/W22/WOO lr = 0.0005`.
- Best in-run `20/class` test accuracy: `77.5%` at epoch `6`.
- Larger eval:
  - `runs/20260522_local_paper_h28_h11_traininternal_best_eval100pc`
  - best checkpoint reached `73.9%` on `100/class` test.

Larger training slice:

- `runs/20260522_local_paper_h28_h11_traininternal_200pc`
- Same architecture and rates, `200/class` train, `50/class` test,
  epochs `10`.
- Best in-run `50/class` test accuracy: `75.8%` at epoch `5`.
- Larger eval:
  - `runs/20260522_local_paper_h28_h11_traininternal_200pc_best_eval100pc`
  - best checkpoint reached `76.3%` on `100/class` test.
- Epoch times after the first epoch were about `21-25s`.

Ablations:

- `runs/20260522_local_paper_h28_h14_traininternal_100pc`
  - `28x28 -> 14x14 -> 40`;
  - best `20/class` test accuracy `75.5%`.
- `runs/20260522_local_paper_h28_h11_trainoutputinternal_100pc`
  - only output-output same-layer learning active;
  - best `20/class` test accuracy `76.5%`.
- `runs/20260522_local_paper_h28_h11_traininternal_100pc_s100`
  - `100/100` sweeps instead of `50/50`;
  - best `20/class` test accuracy `72.5%`.
- `runs/20260522_local_paper_h28_h11_traininternal_r7_100pc`
  - local inter-layer radius `7` instead of `5`;
  - best `20/class` test accuracy `74%`.
- `runs/20260522_local_paper_h28_h11_traininternal_halfinternal_100pc`
  - same-layer learning rates halved to `0.00025`;
  - best `20/class` test accuracy `68.5%`.
- `runs/20260522_local_paper_h28_h11_targetoff0_100pc`
  - off-target label set to `0`;
  - collapsed to a few classes and peaked at `30%`.
- `runs/20260522_local_paper_h28_h11_beta10_100pc`
  - beta `10`;
  - best `20/class` test accuracy `74.5%`.
- `runs/20260522_local_paper_h28_h11_beta25_100pc`
  - beta `2.5`;
  - best `20/class` test accuracy `74.5%`.
- `runs/20260522_local_paper_h28_h11_rep8_100pc`
  - eight output replicas;
  - best `20/class` test accuracy `74.5%`.
- `runs/20260522_local_paper_h28_h11_traininternal_replica_digit_100pc`
  - output shape forced to one class column per digit;
  - best `20/class` test accuracy `74%`.
- `runs/20260522_local_paper_h28_h11_r3_100pc`
  - inter-layer radius `3`;
  - best `20/class` test accuracy `68%`.

Successful continuation:

- `runs/20260522_local_paper_h28_h11_200pc_best_continue_lr001`
  - loaded `runs/20260522_local_paper_h28_h11_traininternal_200pc/best_model.bin`;
  - lowered `W0/W12/W2O lr` from `0.003` to `0.001`;
  - lowered `W11/W22/WOO lr` from `0.0005` to `0.00015`;
  - best `50/class` test accuracy `83.0%`.
- Larger eval:
  - `runs/20260522_local_paper_h28_h11_continue_lr001_best_eval100pc`;
  - `3` free reads reached `80.2%` on `100/class`.
- Second continuation:
  - `runs/20260522_local_paper_h28_h11_continue_lr0005`;
  - loaded the best `lr=0.001` continuation checkpoint;
  - lowered `W0/W12/W2O lr` to `0.0005`;
  - lowered `W11/W22/WOO lr` to `0.000075`;
  - best `50/class` test accuracy `82.8%`.
- Larger eval of the second continuation:
  - `runs/20260522_local_paper_h28_h11_continue_lr0005_best_eval100pc`:
    `80.7%` with `3` free reads.
  - `runs/20260522_local_paper_h28_h11_continue_lr0005_best_eval100pc_reads10`:
    `87.4%` with `10` free reads.
  - `runs/20260522_local_paper_h28_h11_continue_lr0005_best_eval100pc_reads30`:
    `87.9%` with `30` free reads.
- Added `ISING_MNIST_LOCAL_PAPER_EVAL_MODE=mean_reads` to test averaging
  class scores over independent initial conditions instead of selecting the
  lowest-energy read. On the second-continuation checkpoint:
  - mean reads, `10` reads, `50` sweeps: `79.8%`;
  - mean reads, `30` reads, `50` sweeps: `84.2%`;
  - mean reads, `10` reads, `75` sweeps: `82.6%`;
  - lowest-energy read, `10` reads, `75` sweeps: `86.6%`;
  - lowest-energy read, `30` reads, `75` sweeps: `88.2%`.
- Better continuation with more training reads:
  - `runs/20260522_local_paper_h28_h11_continue_lr0005_reads5`;
  - loaded `runs/20260522_local_paper_h28_h11_continue_lr0005/best_model.bin`;
  - used `5/5` free/nudged training reads;
  - kept `W0/W12/W2O lr = 0.0005`;
  - kept `W11/W22/WOO lr = 0.000075`;
  - best `50/class` test accuracy `89.0%` after one epoch.
- Larger eval of the 5-read-training checkpoint:
  - `runs/20260522_local_paper_h28_h11_reads5_best_eval100pc_r30_s75`:
    `91.0%` on `100/class`, `30` reads, `75` sweeps.
  - `runs/20260522_local_paper_h28_h11_reads5_best_eval200pc_r30_s75`:
    `89.0%` on `200/class`, `30` reads, `75` sweeps.
- A further lower-rate continuation:
  - `runs/20260522_local_paper_h28_h11_reads5_continue_lr00025`;
  - best `50/class` test accuracy `87.6%`;
  - did not improve over the 5-read-training checkpoint.
- More-data continuation:
  - `runs/20260522_local_paper_h28_h11_500pc_continue_lr0002_reads5`;
  - loaded `runs/20260522_local_paper_h28_h11_continue_lr0005_reads5/best_model.bin`;
  - trained on `500/class`;
  - used `5/5` free/nudged training reads;
  - used `W0/W12/W2O lr = 0.0002`;
  - used `W11/W22/WOO lr = 0.00003`;
  - best `50/class` validation accuracy `87.8%`;
  - larger eval
    `runs/20260522_local_paper_h28_h11_500pc_best_eval200pc_r50_s75`
    reached `89.7%` on `200/class`, `50` reads, `75` sweeps.
- Current best larger-data continuation:
  - `runs/20260522_local_paper_h28_h11_1000pc_continue_lr0001_reads5`;
  - loaded the `500/class` best checkpoint;
  - trained on `1000/class`;
  - used `5/5` free/nudged training reads;
  - used `W0/W12/W2O lr = 0.0001`;
  - used `W11/W22/WOO lr = 0.000015`;
  - larger eval
    `runs/20260522_local_paper_h28_h11_1000pc_best_eval200pc_r50_s75`
    reached `90.65%` on `200/class`, `50` reads, `75` sweeps.

Interpretation:

- Trainable same-layer couplings are better than the earlier fixed random
  intra-layer couplings for the local architecture.
- The current best local large-eval result moved from `70.6%` to `91.0%` on
  `100/class`, and `90.65%` on `200/class`, when using staged continuation,
  more training data, `5/5` training reads, and lowest-energy inference reads.
- More sweeps, wider local fanout, and weaker same-layer learning did not
  improve the current best run.
- The local architecture now works. The largest remaining practical sensitivity
  is sampling: lowest-energy read selection beats mean-read averaging, and
  `30` inference reads is materially better than the quick `3`-read setting.
