# Simple Time-Averaged XOR

This folder tests the physical `2 -> 4 -> 1` XOR setup with time-averaged
readout statistics and manager-backed reusable workers.

## Current Implementation

`simple_2_4_1_timeavg_learning.jl` now uses a persistent `ProcessManager` for
training and validation. The manager owns the workers and builds per-worker
contexts through `makecontext`, so the resolved composite is not rebuilt for
each worker.

The training path is time-averaged, not only the validation path:

1. run the free phase from a random initial state;
2. restore the free endpoint for the `+β` and `-β` nudged phases;
3. after the normal nudged burn-in, sample Hamiltonian parameter derivatives
   once per configured number of full sweeps;
4. accumulate `sum(dH_plus) - sum(dH_minus)` in the worker-local buffer;
5. after the batch, scale by `2β * n_trajectories * n_derivative_samples`.

Validation still averages the scalar output spin after a burn-in period.

## Runs So Far

Smoke test:

```text
epochs=2, Minit=1, eval_repeats=1, free/nudged=20/20,
train_average_sweeps=2, eval burn-in/average=2/4
```

This verifies that the manager, worker contexts, derivative averager, and plots
all run end to end.

Moderate run:

```text
epochs=800, Minit=4, eval_repeats=8, free/nudged=600/600,
train_average_sweeps=20, beta=2.0, lr=0.003, T=0.005, stepsize=0.4
```

Best behavior was still only `0.75` accuracy with MSE around `0.87`. One XOR
case stayed close to zero or with the wrong sign.

Colder/stronger run:

```text
epochs=1200, Minit=8, eval_repeats=16, free/nudged=1200/1200,
train_average_sweeps=30, beta=4.0, lr=0.002, T=0.002, stepsize=0.8
```

This did not improve the result. The gradient became very small and the run
remained unstable across cases.

## Interpretation

Time averaging the derivative is implemented and functioning, but the first
recipes tried here have not reproduced the earlier successful small scalar XOR
result.

Important naming note: the earlier successful small setup was `2 -> 2x2 -> 1`.
In the code and several docs this appears as `2 -> 4 -> 1`, because the hidden
`2x2` layer is represented as four hidden spins. This time-averaged file is
therefore testing the same small topology, not a different one.

The old endpoint-gradient recipe that worked used roughly:

```text
T = 0.005
stepsize = 0.4
β = 2.0
lr = 0.002
weight_decay = 0
Minit = 8 or 16
free/nudged = 300/300 or 600/600
```

The current averaged-derivative runs have not yet matched that recipe exactly,
so the next step should be to retry the time-averaged-gradient variant with the
known-good small-topology settings before drawing a conclusion about the method.

## Grid Search On May 13

The compact grid was run with the exact known-good scalar settings as the
center point:

```text
T = 0.005
stepsize = 0.4
β = 2.0
lr = 0.002
weight_decay = 0
Minit = 8
free/nudged = 300/300 or 600/600
```

The best result was:

```text
run: ext/IsingLearning/experiments/simple_timeavg_xor/runs/timeavg_grid_20260513_162856/f300_n300_avg1_lr002
epoch: 1600
MSE: 0.117580
accuracy: 1.0
mean outputs: [-0.5690, 0.6856, 0.6567, -0.7396]
```

This is a real learned solution: all four signs are correct and the averaged
readout has reasonable margins. It is still weaker than the earlier endpoint
gradient result, which reached MSE around `0.05-0.07`.

The tested variants did not beat the best run:

| change | best observed effect |
|---|---|
| `600/600` instead of `300/300` | still learns, but best MSE was around `0.18-0.19` |
| derivative time average `train_avg=5` | still learns, best around `0.18` for `600/600` |
| derivative time average `train_avg=20` | worse in the tested runs |
| more epochs to `3000` | plateaued around `0.13-0.17`, no clear improvement |
| `Minit=16` | still learns, best around `0.137`; no improvement over `Minit=8` |
| lower learning rate `0.0015` | still learns, best around `0.129`; slower and not better |
| lower training temperature `0.003` or `0.002` | worse than `T=0.005` |
| stronger clamping `β=3` | much worse; often collapsed weak output means |
| colder validation temperature only | did not improve the averaged-readout MSE |
| initial weight scale `0.25` or `0.12` | did not improve; best was around `0.144` and `0.231` |
| longer nudged phase `300/600` | learned, but best was around `0.134` |
| weaker clamping `β=1.5`, `300/450` | learned, best around `0.125`; close but not better |
| stronger clamping `β=2.5`, `300/450` | poor; output means collapsed near zero |
| shorter free/longer nudged `200/500` | learned, but best was around `0.186` |

The main conclusion is precise: time-averaged validation is useful and the
manager-backed experiment can learn the small XOR, but averaging the EqProp
parameter derivative over many nudged samples is not automatically better here.
For this small graph, the best setting is effectively one derivative sample
after nudged burn-in (`train_avg=1`). Larger derivative averages damp or bias
the useful transient response.

The runs after this conclusion tried the obvious margin-improvement ideas:
longer nudged relaxation, nearby clamping values, colder evaluation, and
different initialization scales. None crossed the `<0.1` MSE target. The best
time-averaged readout remains the exact known-good recipe with `300/300`,
`β=2`, `lr=0.002`, `T=0.005`, and `train_avg=1`.

The multiple-initial-state average is still implemented as separate managed
jobs. That is correct and parallel. If we later want the exact layout where one
worker reruns several initial states and stores a local average, that should be
an engineering change to reduce synchronization overhead, not a change in the
statistical estimator.
