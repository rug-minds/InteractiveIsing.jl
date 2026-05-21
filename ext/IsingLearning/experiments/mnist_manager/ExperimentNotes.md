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
