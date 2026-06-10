# MNIST Agent Instructions

These instructions apply to all MNIST experiments. Also read and follow `agentinstructions.md` in this folder.

## General Grid Knobs

Before a broad grid, run a short diagnostic that reports batch time, acceptance or rejection rates, train/test accuracy, loss, prediction counts, gradient norms, and any sampler or gradient-quality metrics. Do not conclude from one batch unless the run is only a mechanical smoke test.

Useful knobs to vary in MNIST learning grids:

- Nudging strength: beta values below, near, and above the current working value. Include at least one lower-beta and one higher-beta setting when gradient quality is suspect.
- Nudging mode: one-sided, symmetric, forward diagnostic, and any gradient-quality filtered variant being tested.
- Gradient-quality threshold: cosine cutoffs such as `0.0`, `0.5`, `0.8`, `0.95`, `0.99`, and an unfiltered control. Remember that `cos=0.5` is already a `60 deg` disagreement.
- Relaxation sweeps: free-phase sweeps and nudged-phase sweeps. Test whether `500`, `1000`, and `1500` full sweeps materially change phase agreement before assuming equilibration is adequate.
- Nudged initialization: whether the nudged phase starts from the free equilibrium state or from an independently reset state. For EP-style estimates, explicitly document this.
- Temperature: free temperature, nudged temperature, and any temperature annealing or reverse-annealing schedule between phases.
- Langevin stepsize: test smaller and larger stepsizes around the current setting. If phase agreement is poor, include smaller stepsizes rather than only increasing sweeps.
- Target scale: positive and negative target weights for replicated outputs, including whether output-bias priors are projected.
- Learning rate: Adam learning rate, schedule, and any mid-run changes. Do not change LR and beta in the same comparison unless the grid is explicitly factorial.
- Weight decay: fixed decay, adaptive decay, target weight norms, and separate recurrent/input projection decay if available.
- Batch shape: batch size, chunk size, number of workers, selected samples per class, and whether a partial final batch is included.
- Checkpoint state: from-scratch versus resumed runs. If resuming, record the exact checkpoint and whether optimizer state is reset.
- Random seed and data subset: keep seeds fixed for controlled comparisons, then repeat promising settings across seeds.
- Sampler diagnostics: acceptance, phase displacement, free-plus-minus RMS distances, gradient angle/cosine, and skipped-gradient fraction.

When a run learns, write the exact settings and what made the run work into the run folder. When a grid fails, record the negative result with enough detail to avoid repeating the same uninformative sweep.
