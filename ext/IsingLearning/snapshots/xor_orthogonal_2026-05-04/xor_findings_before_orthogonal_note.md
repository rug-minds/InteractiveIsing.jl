# XOR Findings

Short notes from the XOR debugging runs so we do not repeat the same sweeps.

## Training Path

- The active path is `init_mnist_trainer -> _worker_process -> Forward_and_Nudged -> contrastive_gradient -> Optimisers.update`.
- The old `ComputeGradients.jl`, `ChainRules.jl`, and `ep_train_step!` path was not used by these runs.
- Input is written by `apply_input`, which turns off layer 1 in the `ToggledIndexSet` and writes the 16-pattern vector into `state(graph[1])`.
- Output is evaluated by comparing the output vector against the false/true target patterns using dot products.
- In the `Processes` DSL, the dynamics graph must be passed as `model`, and repeated dynamics must be rebound as `model = @repeat ... dynamics()`. The assignment receives the repeated algorithm output fields; it is not a literal capture of the whole return object.
- `ext/IsingLearning/examples/xor_manual_step_debug.jl` bypasses `Process` scheduling and directly calls `Processes.step!` on the sampler context for each free/plus/minus phase. This is now the cleanest scheduling-independent debugging path.

## Gradient Sign

- `Optimisers.update` subtracts the supplied gradient from parameters.
- Laborieux et al. define the symmetric EP estimator in the primitive-function convention as `(∂Φ(sβ)/∂θ - ∂Φ(s-β)/∂θ)/(2β)`. With their nudging convention, this is the update direction, i.e. the negative loss gradient.
- Our relaxation minimizes `H + βC`, so `Φ = -H`.
- Therefore the loss gradient that should be passed to `Optimisers.update` is `(∂H(sβ)/∂θ - ∂H(s-β)/∂θ)/(2β)`.
- This is the current core `contrastive_gradient` sign.
- Scalar analytic check against the actual code conventions: for `H=s²-bs`, `C=1/2(s-y)²`, `b=0.2`, `y=1`, `β=0.01`, the code gives `dL/db ≈ -0.450011`; exact `dL/db = -0.45`.
- The bespoke-script sign flip (`ISING_XOR_SEARCH_GRADIENT_SIGN=-1`) lowered deterministic XOR MSE in some finite-relaxation runs, but that is not evidence that the converged-equilibrium EP sign is wrong. It is more likely a transient / nonconvergence / architecture effect.

## Local Potentials

- For bounded continuous states `StateSet(-1, 1)` with `adjusted=true`, local potentials are not needed for boundedness.
- A grid with local potentials `0.25, 1.0, 4.0` and shallow double wells did not improve XOR separation under bounded adjusted Langevin.
- For adjusted bounded XOR, use `local_potential=0` and `double_well=0` as the clean baseline.
- For unbounded Langevin, local potentials are still required to prevent blow-up.

## Temperature

- Different validation seeds created fake before/after jumps. Use fixed validation seeds or average repeated validation runs.
- `T=0` removes stochastic sampling noise and gives clearer sign diagnostics, but it can also get stuck.
- With the mathematically correct core sign, `T=0` worsened MSE in the tested XOR setup. Treat this as a relaxation/architecture failure, not a sign proof.
- With the flipped bespoke sign, `T=0` gave monotonic MSE decrease but still did not solve XOR. This is useful empirically but should not be moved into the core gradient rule.
- Small nonzero temperature helps exploration but makes metrics noisy unless evaluation is repeated.

## Langevin Variants

- `LocalLangevin(adjusted=true)` is the safest Boltzmann-correct bounded sampler, but it is slow.
- `GlobalLangevin(adjusted=true)` is usually too reject-heavy on bounded high-dimensional states.
- `BlockLangevin(adjusted=true)` is promising for speed, but it had a masking bug: it cached active spins from init and moved clamped input units after `apply_input`.
- That bug was fixed by recomputing active spins inside `BlockLangevin.step!`.

## Current XOR Failure Mode

- With zero local potential and symmetric output patterns, the system strongly tends toward odd symmetry.
- `(false,false)` and `(true,true)` are opposite input patterns but share the false label; one of those even-parity cases tends to stay wrong.
- Small random initial magnetic-field bias lowers baseline MSE, but did not immediately fix 3/4 accuracy in the short runs.
- Next useful knobs are symmetry breaking, readout/target encoding, and better convergence diagnostics before doing larger MNIST runs.
- The XOR search script now restores the best epoch by validation accuracy first, with MSE only as a tie-breaker. The previous MSE-first rule could discard a higher-accuracy epoch.
- The search script's per-config training seed is now controlled by `ISING_XOR_SEARCH_BASE_SEED`; the effective seed is `BASE_SEED + config_index`. This matters because stochastic Metropolis/discrete runs can show transient 4/4 validation for one seed and not another.
- A long pre-patch Metropolis sweep printed one transient `acc=1.0` epoch for `hidden=(16,)`, `out=4`, `bias_scale=0.2`, `lr=0.05`, but rerunning the isolated config after the DSL/restore fixes did not reproduce it. Treat that hit as stochastic/non-reproducible until a fresh run can preserve it.

## Output Clamping

- Direct `Clamping` implements `β/2 * sum_i (s_i - y_i)^2`, so it is only faithful when the task loss is direct squared error on output spin values.
- For a mapped output such as `z = w' * s_out`, the correct squared-error nudging term is `β/2 * (z - y)^2`.
- This creates cross terms between output spins, so block/global proposals need a custom `MultiSpinProposal` `ΔH`; decomposing into independent spin-MSE terms is not equivalent.
- `LinearReadoutClamping` now exists in `ext/IsingLearning/src/ReadoutClamping.jl` for this extension-only experiment.
- A short seed-1 run with `hidden=(16,)`, `out=2`, `bias_scale=0.1`, `relaxation=50` produced an inverted readout for both direct pattern clamping and `LinearReadoutClamping`. That points away from the new readout Hamiltonian as the sole cause.
- Inspired by Laydevant et al. 2021, `ConstantLinearReadoutNudge` now supports a frozen free-phase readout error:
  `H = -β * (target - free_score) * (w' * s_out)`.
- On the current `hidden=(32,)`, `out=8`, bounded continuous baseline, constant nudging did not improve validation accuracy. A small `β in (0.1, 0.5, 1.0)` and `lr in (0.01, 0.02)` sweep still restored the pre-training `0.75` solution as best.

## Architecture Notes

- The XOR examples use `AllToAllWeightGenerator` between adjacent layers, so the hidden layer is fully connected to the input layer and the output/readout layer is fully connected to the previous layer. There are no same-layer hidden connections and no input-output skip connection in these architectures.
- The custom XOR weight generator uses `randn`, so initial weights include both positive and negative values. The default `ReducedBoltzmannArchitecture` weight generator is different: if no generator is passed, it creates positive unit weights.
- Example-local weight normalization was added with `ISING_XOR_SEARCH_WEIGHT_NORM` and `ISING_XOR_MANUAL_WEIGHT_NORM`. It rescales the signed weight vector to a target RMS after each optimiser update; it does not clip signs or force weights positive.
- Best practical baseline so far: `hidden=(32,)`, `out=8`, bounded continuous states, adjusted local Langevin, no local potential, no double well.
- `hidden=(64,)` did not improve XOR accuracy in the tested runs; it mostly increased parameter movement and changed margins.
- `out=8` was better than `out=16` in the bounded adjusted no-local-potential runs.
- `out=2` can show apparent jumps between 0.25/0.5/0.75 accuracy depending on validation seed, so it is noisy and not a reliable readout for judging learning.
- `out=8` consistently starts around 3/4 accuracy and lowers MSE slightly, but still has one parity case wrong.
- After the `model = @repeat ...` DSL fix, an accuracy-first block-Langevin sweep over `hidden=(32,)`, `out=8`, `bias_scale in (0.05, 0.1, 0.2)`, and three weight seeds still plateaued at 0.75 accuracy. Best MSE was about `1.3538`.
- A discrete Metropolis probe with `hidden=(32,)`, `out=8`, `bias_scale=0.1`, flipped debug sign, and `relaxation_steps=300` also topped out at 0.75 and had much worse score MSE.
- Manual stepping confirms the plateau is not just process scheduling. A deterministic manual `hidden=(32,)`, `out=8`, `constant_readout_hterm`, block-Langevin run with random init stayed at `0.5` in a 2-epoch smoke. With `init=ones`, it improved to `0.75` and MSE about `0.94`, but collapsed outputs close to all ones.
- Manual discrete Metropolis with `hidden=(16,)`, `out=4`, `init=ones`, `bias_scale=0.2`, `lr=0.05`, `β=0.1` also topped out at `0.75`.

## Manual Context Stepping

- `xor_manual_step_debug.jl` now uses `LayeredIsingGraphLayer` and `init_mnist_trainer` to build the same worker graph, `_state` buffers, and sampler context as the threaded path.
- The manual loop now keeps the full worker `ProcessContext`, extracts the resolved identifiable `@dynamics` wrapper from the worker algorithm, and steps that wrapper directly.
- The step call is `Processes.step!(dynamics_stepper, context, Processes.Unstable())`; the identifiable wrapper creates the context view and merges the return fields back into the full context.
- `Unstable()` is intentional here because the first Langevin step adds diagnostics such as `acceptance_rate` and changes the initialized `group_steps` field from the config `Ref` into the scalar returned by `step!`.
- This gives a scheduling-independent path while still using the real context initialization, input buffers, target buffers, and graph parameter sync helpers.
- A one-epoch smoke with `relax=2` completed and showed nonzero gradients/updates, so the manual path is currently compiling and stepping through real dynamics contexts.

## Weight-Normalized Architecture Probe

Short block-Langevin survey: `constant_readout_hterm`, bounded continuous states, `bias_scale=0.1`, `lr=0.02`, `β=0.1`, `relaxation_steps=50`, `epochs=15`, `weight_norm=0.05`, `weight_seed=2`, `bias_seed=11`.

- `hidden=(32,)`, `out=8`: best of the probe, `0.75` accuracy, score MSE about `1.3637`; best epoch was still epoch 0.
- `hidden=(64,)`, `out=8`: `0.75` accuracy, score MSE about `1.3758`; not better than hidden 32.
- `hidden=(16,)`, `out=8`: `0.75` accuracy, score MSE about `1.3984`.
- `out=4` with hidden 16/32/64 stayed at `0.5` accuracy.
- Deeper `hidden=(32,16)` was worse in this probe: `out=4` reached only `0.25`, and `out=8` also stayed at `0.25`.
- Weight normalization controlled the RMS exactly as intended, but it did not by itself create a learning improvement in this short sweep.

## Paper-Inspired XOR Changes

- The 2026 architecture paper trains XOR for up to 20,000 epochs and uses `Minit = 5`, i.e. five random initial steady states per data sample per epoch, to handle multistability.
- `ISING_XOR_SEARCH_MINIT` and `ISING_XOR_MANUAL_MINIT` were added to mirror this averaging in the local XOR scripts.
- Scalar output (`out=1`) is now supported and is closer to the paper's single-output-node XOR setup.
- `ISING_XOR_SEARCH_SKIP_WEIGHT_SCALE` adds direct input-output skip weights as an XOR-only architecture experiment. This was inspired by the paper's conclusion that higher-connectivity / skip-like lattices train better.
- A critical issue with the original global-pattern XOR input is symmetry: `(false,false)` and `(true,true)` are exact negatives but have the same label. Without an additional constant feature, the trained/evaluated response tends to remain odd under `x -> -x`, causing the persistent 0.75 ceiling.
- `ISING_XOR_SEARCH_INPUT_BIAS=true` appends one clamped constant input unit. With `hidden=(8,)`, `out=1`, `Minit=5`, `weight_norm=0.2`, `β=0.1`, `relax=100`, `weight_seed=2`, and base seed `21000`, the process-based search classified all four XOR inputs (`acc=1.0`) from the first epoch.
- This is a representability / architecture success, not yet a robust learning success: a small seed sweep with `INPUT_BIAS=true` still produced 0.5 accuracy for several seeds. The current working setup has low margins, so the next target is robustness rather than mere existence.
- The lower-error run uses `readout_hterm` rather than direct spin-pattern clamping. Exact rerun:
  `hidden=(16,)`, `out=1`, `readout_target=0.2`, `INPUT_BIAS=true`, `Minit=5`, `weight_norm=0.2`, `lr=0.01`, `β=0.1`, `relax=100`, `BlockLangevin(adjusted=true)`, `weight_seed=2`, `bias_seed=11`, `base_seed=23000`.
  It kept `acc=1.0` and lowered readout score MSE from `0.051938` to `0.045922` by epoch 450. Final outputs were approximately:
  false/false `-0.114`, false/true `0.591`, true/false `0.079`, true/true `-0.106` for targets `-0.2, 0.2, 0.2, -0.2`.
- That `acc=1.0` is not stable under heavier averaging over validation initial states. With the same untrained weights and `EVALUATION_REPEATS=50`, the averaged zero-epoch result was `acc=0.75`, score MSE `0.040384`, and one true case had negative score. So the earlier `acc=1.0` was a low-repeat/seed-dependent classification result, not an "all initial states classify correctly" result.
- This explains why earlier `acc=1.0` did not imply minimal MSE: classification only needs the readout sign to be correct, while MSE also penalizes the score magnitude relative to the chosen target scale. `readout_target=0.2` is much better matched to the current output magnitudes than the old scalar pattern score target near `±2`.
- A 10,000 epoch curve was saved from the same random signed `J` setup using `EVALUATION_REPEATS=5` and `LOG_EVERY=100`.
  The MSE decreased from `0.051938` before training to a restored best of `0.033841` at epoch 5100. The final logged epoch 10000 had MSE `0.037398`, while best-epoch restoration returned the epoch-5100 parameters.
  Accuracy was not robust throughout the run: it stayed at `1.0` for long stretches, dropped to `0.75` late in training, and returned to `1.0` at the final logged epoch. This supports optimizing readout MSE, but not yet robust classification across relaxation seeds.
  Artifacts: `ext/IsingLearning/runs/xor_10k_20260504_202044/xor_10k_mse.png` and `xor_10k_mse.csv`.

## Bias / Symmetry Breaking

- Zero initial bias keeps a strong symmetry and gives deterministic `T=0` MSE around `1.19` for `hidden=(32,)`, `out=8`.
- Small random initial magnetic-field bias improves the baseline MSE:
  - `bias_scale=0.05`, short run, `relax=100`: MSE around `0.943`, still 3/4 accuracy.
  - Longer `relax=1000` runs with flipped sign still stayed at 3/4, but larger bias plus higher LR improved MSE.
- In the current longer sweep, `hidden=(32,)`, `out=8`, `bias_scale=0.1`, `lr=0.02`, flipped sign reached the best seen MSE so far, about `1.1409` at epoch 100, then overtrained.
- Higher LR (`0.02`) can improve faster but overtrains after roughly 50-100 epochs in these runs.

## Best Completed Sweep

Command family: bounded continuous XOR, adjusted local Langevin, `T=0`, no local potential, no double well, flipped bespoke gradient sign.

- Best config:
  - `hidden=(32,)`
  - `out=8`
  - `bias_scale=0.1`
  - `lr=0.02`
  - `beta=0.05`
  - `relaxation_steps=1000`
  - `stepsize=0.005`
- Result:
  - Before: MSE `1.192853`, accuracy `0.75`
  - Best epoch: `100`
  - Best/after: MSE `1.140863`, accuracy `0.75`, margin `0.372569`
- Ranking trend:
  - Hidden 32 beat hidden 64.
  - Higher bias improved hidden 32.
  - `lr=0.02` beat `lr=0.005`, but only with best-epoch restoration because later epochs overtrained.
  - None of the tested configs reached 4/4 XOR classification.
