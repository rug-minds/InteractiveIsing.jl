# 2-Input Continuous Langevin XOR Success

This note records the first continuous Langevin run that matched the practical
success criterion on honest two-input XOR.

It does not replace the discrete Metropolis baseline. It shows that Langevin
can solve the same task when the initialization and effective-temperature
regime are chosen correctly.

## Exact Run

```bash
ISING_XOR_2IN_STATE=continuous \
ISING_XOR_2IN_DYNAMICS=langevin \
ISING_XOR_2IN_INIT_MODE=zero \
ISING_XOR_2IN_EPOCHS=2000 \
ISING_XOR_2IN_LOG_EVERY=250 \
ISING_XOR_2IN_MINIT=4 \
ISING_XOR_2IN_EVAL_REPEATS=32 \
ISING_XOR_2IN_FREE_RELAXATION=1000 \
ISING_XOR_2IN_NUDGED_RELAXATION=1000 \
ISING_XOR_2IN_WEIGHT_SCALE=0.2 \
ISING_XOR_2IN_BIAS_SCALE=0.05 \
ISING_XOR_2IN_TEMP=0.001 \
ISING_XOR_2IN_BETA=1.0 \
ISING_XOR_2IN_LR=0.01 \
ISING_XOR_2IN_WEIGHT_DECAY=0 \
ISING_XOR_2IN_STEPSIZE=0.1 \
ISING_XOR_2IN_BLOCK_SIZE=8 \
ISING_XOR_2IN_WEIGHT_SEED=13 \
ISING_XOR_2IN_BIAS_SEED=23 \
ISING_XOR_2IN_BASE_SEED=87300 \
ISING_XOR_2IN_DIR=ext/IsingLearning/runs/xor_2input_langevin_zero_T0001_s010_lr001_nodecay \
julia --project=ext/IsingLearning ext/IsingLearning/examples/xor_statistical_ep_2input.jl
```

Artifacts:

```text
ext/IsingLearning/runs/xor_2input_langevin_zero_T0001_s010_lr001_nodecay/
```

## Result

Logged best point:

```text
epoch 1500: MSE 0.0268128541, accuracy 1.0
```

Restored best checkpoint reevaluation:

```text
MSE 0.038118, accuracy 1.0
```

Restored output means:

```text
(false, false) -> [ 0.9576, -0.9585]
(false, true)  -> [-0.8851,  0.9788]
(true, false)  -> [-0.8065,  0.5428]
(true, true)   -> [ 0.9090, -0.8182]
```

The `true,false` case is still the weakest and has significant variance, but
the result is below the `0.1` MSE target and classifies all four cases
correctly.

## Simple Experiment File Check

The simpler standalone file now reproduces the same result:

```text
ext/IsingLearning/experiments/simple_langevin_xor/simple_2_16_2_locallangevin.jl
```

Default run checked on 2026-05-17:

```text
architecture = 2 -> 16 -> 2
dynamics = BlockLangevin(adjusted=false, stepsize=0.1, block_size=8)
init_mode = zero
T = 0.001
β = 1.0
lr = 0.01
weight_decay = 0
free/nudged = 1000/1000
Minit = 4
weight_seed = 13
bias_seed = 23
base_seed = 87300
```

It reached MSE `0.000124`, accuracy `1.0` by epoch `250`, and finished at
MSE `0.000123`, accuracy `1.0` at epoch `2000`.

The important fix was to remove the experiment-local plus/minus wiring that
used runtime `clamping_beta` and separate hand-captured plus/minus branches.
The file now uses the same proven shape as the working example:

```text
forward()
nudged.algorithm()
contrastive_gradient(free_model, plus_capture, minus_capture, β)
```

That kept the gradient collection aligned with the shared trainer path.

## What Changed Relative To Failed Langevin Runs

The important differences were:

- `INIT_MODE=zero`, not random continuous initial states;
- `T=0.001`, much colder than the failed `T=0.5` matched Metropolis-temperature
  probe;
- `stepsize=0.1`, not `0.05`;
- `lr=0.01`, not `0.0015`;
- no weight decay;
- best-checkpoint restoration, because the run degraded after the good point.

The failed matched Langevin probe used the successful Metropolis temperature
`T=0.5`, random initial states, `stepsize=0.05`, `lr=0.0015`, and weight decay.
That was not an equivalent effective-temperature regime for continuous bounded
states. It produced weak output means near zero and high output variance.

## Interpretation

For the continuous bounded system, the relevant scale is not `T` alone. It is
the effective temperature relative to the learned fields:

```text
effective noise scale ~ T / typical |J|
```

The bad continuous runs had output standard deviations around `0.5` to `0.9`
and output means near zero. That means the states were not in a crisp class
regime, even when the sign of the mean briefly gave good accuracy.

The successful run used colder dynamics and allowed the weights to grow more
aggressively. That moved the system into a regime where most output means are
near the target corners.

Zero initialization also mattered. With random continuous initial states, the
same style of run found partial solutions but stayed much noisier. For this
small all-to-all test, zero initialization removes unnecessary basin variance
and lets the contrastive signal train a consistent input-to-output map.

## Caveats

- This is still seed/config sensitive.
- The final epoch was worse than the best epoch, so learning-rate control or
  early stopping is needed.
- This uses unadjusted `BlockLangevin`; it is a practical learning dynamics,
  not an exact Boltzmann sampler claim.
- The `true,false` output variance is still high enough that more validation
  repeats and seed checks are needed before calling this robust.
