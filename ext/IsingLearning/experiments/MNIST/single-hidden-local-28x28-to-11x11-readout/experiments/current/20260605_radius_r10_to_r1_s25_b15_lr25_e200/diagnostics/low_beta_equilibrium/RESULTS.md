# Low-Beta Equilibrium Diagnostic Results

Run time: 2026-06-05 around 14:02-14:06.

Checkpoint: `../../r8/best_params.bin`

Screening settings:

- samples: 10 balanced test samples, 1 per class
- repeats: 1 and 4 independent random initial states
- free burn-in: 25 sweeps
- output averaging: 10 sweeps, sampled every sweep
- nudged beta values: 0.05, 0.1, 0.25, 0.5

Files:

- `low_beta_multi_init_timeavg_summary.csv`
- `low_beta_multi_init_timeavg_samples.csv`
- `low_beta_multi_init_timeavg_summary.png`
- `low_beta_output_trace_logger.csv`
- `low_beta_output_trace_logger.png`

Initial observations:

- Free readout is still poor under this diagnostic: 0.10 accuracy for both 1-repeat and 4-repeat averaged readout.
- Multi-init averaging improves the free target margin from `-0.215` to `-0.130`, but does not recover classification by itself on this small subset.
- Low beta is not invisible to the system:
  - beta 0.05 gives mean target shift `0.185` with 1 repeat and `0.541` with 4 repeats.
  - beta 0.1 gives mean target shift about `0.81` and makes all 10 nudged predictions target-aligned in this diagnostic.
  - beta 0.25 and 0.5 only marginally increase the shift beyond beta 0.1.
- The nudged rows should not be interpreted as classifier accuracy, because the target field is applied. They measure target response strength and output relaxation under a low-beta nudge.
- The trace logger wrote 3 samples x 2 repeats x free/nudged trajectories. The plotted statistic is max class score over logged sweeps; the CSV contains all ten class scores.

Implication for next training test:

The useful next variant is not simply "raise beta". Try a low-beta training estimator that averages equilibrium states over several initializations and/or time samples before forming the contrastive gradient. The current single best/final state readout is likely too noisy for the low-beta numerator.

## State-Averaged Training Smoke Runs

Run time: 2026-06-05 around 14:34-14:38.

Implementation:

- Script: `low_beta_stateavg_training.jl`
- Launcher: `launch_stateavg_training_diagnostics.ps1`
- Output root: `stateavg_training_runs/`
- Radius: `r=8`
- Initialization: scratch, no checkpoint resume
- Estimator: one-sided EqProp, `(nudged_avg - free_avg) / beta`
- Not used here: symmetric `(+beta - -beta) / (2beta)`
- Workers: 16
- Dataset: 20 train samples/class, 10 test samples/class
- Epochs: 15
- Burn-in: 15 sweeps per phase
- State averaging: 5 sweeps per phase, sampled once per sweep

Results:

| run | best test accuracy | best epoch | final test accuracy | final train accuracy |
| --- | ---: | ---: | ---: | ---: |
| `r8_stateavg_beta0p10_lr1e-5_burn15_avg5_e15` | 0.18 | 3 | 0.09 | 0.08 |
| `r8_stateavg_beta0p05_lr5e-6_burn15_avg5_e15` | 0.16 | 5 | 0.07 | 0.10 |

Interpretation:

Whole-state time averaging is now wired and runs, but these low-beta one-sided training smokes did not learn. The results stayed near chance and rolled down after a small early fluctuation. This does not rule out stochastic/averaged EqProp, but it means this particular estimator and learning-rate scale are not enough. A fairer next test would be the symmetric `+beta/-beta` estimator, because it removes the free-state subtraction from the numerator and is usually the cleaner finite-difference form at small beta.

## Symmetric State-Averaged Temperature Grid

Run time: 2026-06-05 around 14:46-15:41.

Implementation:

- Script: `low_beta_stateavg_training.jl`
- Launcher: `launch_symmetric_temperature_grid.ps1`
- Output root: `symmetric_temperature_grid/`
- Radius: `r=8`
- Initialization: scratch, no checkpoint resume
- Estimator: symmetric `(+beta_avg - -beta_avg) / (2beta)`
- Workers: 16
- Dataset: 50 train samples/class, 20 test samples/class
- Epochs: 60
- Burn-in: 25 sweeps per phase
- State averaging: 10 sweeps per phase, sampled once per sweep
- Betas: `0.1`, `0.25`
- Cold temperatures: `0.003`, `0.01`, `0.03`
- Reverse-anneal peak temperatures: `0.3`, `1.0`

Results:

| run | best test accuracy | best epoch | final test accuracy | final prediction counts |
| --- | ---: | ---: | ---: | --- |
| `r8_sym_b0p1_Tc0p003_Tr0p3_burn25_avg10_e60` | 0.15 | 7 | 0.10 | `0-0-0-0-0-0-200-0-0-0` |
| `r8_sym_b0p1_Tc0p003_Tr1p0_burn25_avg10_e60` | 0.14 | 6 | 0.10 | `200-0-0-0-0-0-0-0-0-0` |
| `r8_sym_b0p1_Tc0p01_Tr0p3_burn25_avg10_e60` | 0.16 | 9 | 0.10 | `200-0-0-0-0-0-0-0-0-0` |
| `r8_sym_b0p1_Tc0p01_Tr1p0_burn25_avg10_e60` | 0.13 | 2 | 0.10 | `200-0-0-0-0-0-0-0-0-0` |
| `r8_sym_b0p1_Tc0p03_Tr0p3_burn25_avg10_e60` | 0.13 | 5 | 0.10 | `200-0-0-0-0-0-0-0-0-0` |
| `r8_sym_b0p1_Tc0p03_Tr1p0_burn25_avg10_e60` | 0.13 | 1 | 0.10 | `200-0-0-0-0-0-0-0-0-0` |
| `r8_sym_b0p25_Tc0p003_Tr0p3_burn25_avg10_e60` | 0.13 | 11 | 0.10 | `200-0-0-0-0-0-0-0-0-0` |
| `r8_sym_b0p25_Tc0p003_Tr1p0_burn25_avg10_e60` | 0.12 | 0 | 0.10 | `200-0-0-0-0-0-0-0-0-0` |
| `r8_sym_b0p25_Tc0p01_Tr0p3_burn25_avg10_e60` | 0.14 | 2 | 0.10 | `200-0-0-0-0-0-0-0-0-0` |
| `r8_sym_b0p25_Tc0p01_Tr1p0_burn25_avg10_e60` | 0.14 | 11 | 0.10 | `200-0-0-0-0-0-0-0-0-0` |
| `r8_sym_b0p25_Tc0p03_Tr0p3_burn25_avg10_e60` | 0.14 | 4 | 0.10 | `200-0-0-0-0-0-0-0-0-0` |
| `r8_sym_b0p25_Tc0p03_Tr1p0_burn25_avg10_e60` | 0.12 | 10 | 0.10 | `200-0-0-0-0-0-0-0-0-0` |

Interpretation:

The symmetric estimator is wired and executes, but this temperature/beta grid did not recover learning. All settings converged to a degenerate single-class classifier by the end of training. Lower cold temperature, higher cold temperature, lower reverse peak, higher reverse peak, and beta 0.1 vs 0.25 did not prevent collapse in this setup.

The main useful result is negative: the collapse is not just an artifact of the one-sided low-beta estimator. It still appears with symmetric finite differences and full-state time averaging. That points back toward the training dynamics/parameterization producing a strong output-bias attractor, not merely noisy endpoint sampling.

## Observable-Symmetric Low-Beta Grid

Started: 2026-06-05 17:58.

Implementation:

- Script: `low_beta_stateavg_training.jl`
- Launcher: `launch_observable_symmetric_grid.ps1`
- Output root: `observable_symmetric_grid/`
- Radius: `r=8`
- Initialization: scratch, no checkpoint resume
- Estimator: symmetric `(+beta - -beta) / (2beta)` using sampled observables directly
- Gradient observable: average `s_i s_j` during each nudged phase, not product of averaged states
- Workers: 16
- Dataset: 50 train samples/class, 20 test samples/class
- Epochs: 80
- Burn-in: 25 sweeps per phase
- Observable averaging: 10 or 20 sweeps per nudged phase, sampled once per sweep

Results:

| run | best test accuracy | best epoch | final test accuracy | final prediction counts |
| --- | ---: | ---: | ---: | --- |
| `r8_obs_sym_b0p5_Tc0p001_Tr0p1_lr0p00001_avg10_e80` | 0.165 | 24 | 0.10 | `200-0-0-0-0-0-0-0-0-0` |
| `r8_obs_sym_b0p5_Tc0p003_Tr0p1_lr0p00001_avg10_e80` | 0.150 | 4 | 0.10 | `200-0-0-0-0-0-0-0-0-0` |
| `r8_obs_sym_b1p0_Tc0p001_Tr0p05_lr0p000005_avg20_e80` | 0.165 | 26 | 0.10 | `200-0-0-0-0-0-0-0-0-0` |
| `r8_obs_sym_b1p0_Tc0p001_Tr0p1_lr0p00001_avg10_e80` | 0.160 | 7 | 0.10 | `200-0-0-0-0-0-0-0-0-0` |
| `r8_obs_sym_b1p0_Tc0p003_Tr0p1_lr0p00001_avg10_e80` | 0.150 | 14 | 0.10 | `200-0-0-0-0-0-0-0-0-0` |
| `r8_obs_sym_b1p0_Tc0p003_Tr0p3_lr0p00001_avg10_e80` | 0.150 | 20 | 0.10 | `200-0-0-0-0-0-0-0-0-0` |
| `r8_obs_sym_b2p0_Tc0p001_Tr0p1_lr0p00001_avg10_e80` | 0.170 | 10 | 0.10 | `200-0-0-0-0-0-0-0-0-0` |
| `r8_obs_sym_b2p0_Tc0p003_Tr0p3_lr0p00001_avg10_e80` | 0.140 | 15 | 0.10 | `200-0-0-0-0-0-0-0-0-0` |

Interpretation:

The direct observable estimator runs and is fast, but every setting still collapses to a single-class classifier. This rules out the specific `E[s_i]E[s_j]` approximation as the only cause of collapse. Colder equilibrium temperatures, beta up to 2.0, smaller reverse peaks, lower learning rate, and longer observable averaging did not fix the collapse in this scratch r8 setup.

## Observable-Symmetric Higher-Beta Grid

Started: 2026-06-05.

Implementation:

- Script: `low_beta_stateavg_training.jl`
- Launcher: `launch_observable_symmetric_high_beta_grid.ps1`
- Output root: `observable_symmetric_high_beta_grid/`
- Radius: `r=8`
- Initialization: scratch, no checkpoint resume
- Estimator: symmetric `(+beta - -beta) / (2beta)` using sampled observables directly
- Workers: 16
- Dataset: 50 train samples/class, 20 test samples/class
- Epochs: 60
- Burn-in/averaging: 25 burn-in sweeps + 10 observable averaging sweeps for both `+beta` and `-beta`
- Betas tested: `3`, `5`, `10`, `15`

Result:

All higher-beta settings collapsed to chance-level single-class behavior. The best transient seen in this group was about `0.18`, but final accuracies were `0.10`.

Interpretation:

Increasing beta alone did not rescue the symmetric observable estimator. It strengthens the output force but still appears to push the system into the same degenerate output-bias attractor.

## Observable-Symmetric Asymmetric-Sweep Grid

Started: 2026-06-05.

Implementation:

- Script: `low_beta_stateavg_training.jl`
- Launcher: `launch_observable_symmetric_asym_sweeps_grid.ps1`
- Output root: `observable_symmetric_asym_sweeps_grid/`
- Radius: `r=8`
- Initialization: scratch, no checkpoint resume
- Estimator: symmetric sampled observables
- Workers: 16
- Dataset: 50 train samples/class, 20 test samples/class
- Epochs: 60
- Variable: fewer free sweeps and more nudged sweeps, to test whether the nudged equilibrium estimate was the noisy part.

Results:

| run | best test accuracy | best epoch | final test accuracy | final prediction counts |
| --- | ---: | ---: | ---: | --- |
| `r8_obs_sym_b5p0_free10p3_nudge50p25_lr0p00001_e60` | 0.15 | 10 | 0.10 | `200-0-0-0-0-0-0-0-0-0` |
| `r8_obs_sym_b10p0_free10p3_nudge50p25_lr0p00001_e60` | 0.16 | 23 | 0.10 | `200-0-0-0-0-0-0-0-0-0` |
| `r8_obs_sym_b15p0_free10p3_nudge50p25_lr0p00001_e60` | 0.14 | 21 | 0.10 | `200-0-0-0-0-0-0-0-0-0` |
| `r8_obs_sym_b5p0_free5p1_nudge75p50_lr0p000005_e60` | 0.19 | 49 | 0.115 | `27-53-42-4-14-23-24-3-8-2` |
| `r8_obs_sym_b10p0_free5p1_nudge75p50_lr0p000005_e60` | 0.16 | 13 | 0.11 | `8-68-40-5-10-39-15-6-9-0` |

Interpretation:

More nudged averaging plus lower learning rate softened the hard single-class collapse in the last two runs, but it still did not create useful learning. This is evidence that sampling noise and nudged-phase variance contribute to instability, but are not sufficient explanations.

## Tangent-Nudge Observable Grid

Started: 2026-06-05 22:33.

Implementation:

- Script: `low_beta_stateavg_training.jl`
- Launcher: `launch_tangent_observable_grid.ps1`
- Output root: `tangent_observable_grid/`
- Radius: `r=8`
- Initialization: scratch, no checkpoint resume
- Estimator: symmetric sampled observables
- Nudge: tangent force from the free equilibrium output error, applied as `beta * (target - free_output)` during the nudged phase
- Workers: 16
- Dataset: 50 train samples/class, 20 test samples/class
- Epochs: 80

Current status:

| run | status | best test accuracy | best epoch | current/final test accuracy | current/final prediction counts |
| --- | --- | ---: | ---: | ---: | --- |
| `r8_tangent_obs_b3p0_free10p3_nudge50p25_lr0p00001_e80` | complete | 0.15 | 26 | 0.10 | `200-0-0-0-0-0-0-0-0-0` |
| `r8_tangent_obs_b5p0_free10p3_nudge50p25_lr0p00001_e80` | running at last update | 0.16 | 2 | 0.105 at epoch 9 | `30-27-41-6-20-33-16-11-11-5` |

Interpretation:

The first tangent-nudge run did not rescue learning and collapsed by epoch 80. The second run was still running when this note was written, so this section is not final yet.

## Learning-Slope Analysis

Generated with `plot_learning_slopes.jl`.

Outputs:

- `analysis/learning_slopes/learning_slopes_all.csv`
- `analysis/learning_slopes/learning_slope_summary.csv`
- `analysis/learning_slopes/main_radius_accuracy_and_smoothed_slope.png`
- `analysis/learning_slopes/all_runs_test_slope_summary.png`
