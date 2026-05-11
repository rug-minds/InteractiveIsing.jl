# 2-Input Discrete XOR Successful Run

This note documents the first honest two-input XOR run that reached stable
classification and near-zero output-vector MSE.

It is separate from the earlier `xor_successful_learning.md` note. That older
run used a four-way one-hot input code, which proved the learning plumbing but
also made XOR linearly separable at the input. The run documented here uses the
actual two XOR input bits.

## Files And Artifacts

Runnable experiment:

```text
ext/IsingLearning/examples/xor_statistical_ep_2input.jl
```

Successful run directory:

```text
ext/IsingLearning/runs/xor_2input_discrete_T05_seed13/
```

Artifacts in that directory:

```text
xor_statistical_ep_2input.csv
xor_statistical_ep_2input.png
xor_statistical_ep_2input_trained_graph.jld2
```

The trained graph from this run was also loaded into:

```text
examples/XORInteractive.jl
```

That interactive example should use the trained snapshot values, not synthetic
handwritten weights.

## Exact Run

The successful run used:

```bash
ISING_XOR_2IN_STATE=discrete \
ISING_XOR_2IN_DYNAMICS=metropolis \
ISING_XOR_2IN_EPOCHS=5000 \
ISING_XOR_2IN_LOG_EVERY=250 \
ISING_XOR_2IN_MINIT=8 \
ISING_XOR_2IN_EVAL_REPEATS=64 \
ISING_XOR_2IN_FREE_RELAXATION=1000 \
ISING_XOR_2IN_NUDGED_RELAXATION=1000 \
ISING_XOR_2IN_WEIGHT_SCALE=0.2 \
ISING_XOR_2IN_BIAS_SCALE=0.05 \
ISING_XOR_2IN_TEMP=0.5 \
ISING_XOR_2IN_BETA=1.0 \
ISING_XOR_2IN_LR=0.0015 \
ISING_XOR_2IN_WEIGHT_SEED=13 \
ISING_XOR_2IN_BIAS_SEED=23 \
ISING_XOR_2IN_BASE_SEED=82000 \
ISING_XOR_2IN_DIR=ext/IsingLearning/runs/xor_2input_discrete_T05_seed13 \
julia --project=ext/IsingLearning ext/IsingLearning/examples/xor_statistical_ep_2input.jl
```

Defaults that matter but are easy to miss:

```text
ISING_XOR_2IN_RULE=ep
ISING_XOR_2IN_TARGET_FREE_SIGN=target_minus_free
ISING_XOR_2IN_INIT_MODE=random
ISING_XOR_2IN_WEIGHT_DECAY=1e-4
```

`TARGET_FREE_SIGN` is only used by the optional target-free surrogate branch.
The successful run uses the real symmetric EP rule, so that setting is not part
of the gradient that solved the task.

## Encoding

The input is the actual two-bit XOR input:

```text
(false, false) -> [-1, -1]
(false, true)  -> [-1, +1]
(true, false)  -> [+1, -1]
(true, true)   -> [+1, +1]
```

The output is a direct two-spin bipolar class code:

```text
XOR false -> [+1, -1]
XOR true  -> [-1, +1]
```

This is important. The successful run does not hide a nonlinear four-way input
encoding. The hidden layer has to create the XOR separation.

## Graph And Hamiltonian

Architecture:

```text
2 input spins -> 16 hidden spins -> 2 output spins
```

State set:

```text
{-1, +1}
```

Dynamics:

```text
IsingMetropolis()
T = 0.5
```

Hamiltonian terms:

```text
Bilinear + MagField + Clamping
```

There is no local potential and no double well in this successful run. Bounded
states come from the discrete state set, not from a polynomial local term.

The input layer is fixed by the input application during each phase. Hidden and
output spins are sampled. The target is direct output-spin squared error, so the
standard `Clamping` term is the correct nudging term.

## Learning Rule

This run uses the symmetric EP estimator implemented in
`ext/IsingLearning/src/Gradient.jl`.

The sampler minimizes:

```math
H_\beta(\theta, s) = H(\theta, s) + \beta C(s, y).
```

The plus phase samples near a state of `H + βC`; the minus phase samples near a
state of `H - βC`. The gradient passed to `Optimisers.update` is:

```math
\nabla_\theta L
\approx
\frac{\partial_\theta H(\theta, s_\beta)
      - \partial_\theta H(\theta, s_{-\beta})}{2\beta}.
```

`Optimisers.update` applies descent, so this is the correct sign for the code's
Hamiltonian convention.

The successful run did not use the target-free surrogate gradient. That branch
was useful as a diagnostic, but it was not the rule that solved this run.

## Phase Initialization

For each sample and repeated initial state, the phases are:

```text
random state
-> apply the two-bit input
-> free relaxation
-> store the free relaxed state
-> restore that free state, apply input and target, set +β
-> plus/nudged relaxation
-> restore the same free state, apply input and target, set -β
-> minus/anti-nudged relaxation
-> contrastive gradient from plus minus minus
```

So the clamped phases do start from the free relaxed state. They are not started
from fresh random states.

The validation metric is also statistical. It runs repeated free relaxations
from fixed validation seeds and averages the output spins before computing MSE
and accuracy.

## Result

CSV highlights:

```text
epoch 0:    MSE 1.1197509766, accuracy 0.25
epoch 500:  MSE 0.0083007813, accuracy 1.0
epoch 1250: MSE 0.0019531250, accuracy 1.0
epoch 3500: MSE 0.0002441406, accuracy 1.0
epoch 4000: MSE 0.0002441406, accuracy 1.0
epoch 5000: MSE 0.0018310547, accuracy 1.0
```

Representative restored/best output means:

```text
(false, false) -> [ 0.9375, -0.9375]
(false, true)  -> [-0.9375,  0.9375]
(true, false)  -> [-0.9688,  0.9688]
(true, true)   -> [ 0.9688, -0.9688]
```

The trained interactive snapshot also passed a headless deterministic check in
which all four cases relaxed to the correct sign pattern after a longer
Metropolis run:

```text
(false, false) -> [ 1, -1]
(false, true)  -> [-1,  1]
(true, false)  -> [-1,  1]
(true, true)   -> [ 1, -1]
```

## What Made This Work

The important changes were not just "more epochs".

1. **Use the real two-bit input**

   Four-way one-hot input is useful as a plumbing test, but it makes XOR
   linearly separable before the Ising system sees it. The successful run uses
   only the two original input bits.

2. **Use discrete spins for this discrete problem**

   The continuous Langevin versions repeatedly got stuck with weak outputs or
   collapsed to partial solutions. Discrete spins let the model represent the
   desired class states directly as `-1` and `+1`.

3. **Use finite, not near-zero, temperature**

   Very low temperature froze the discrete system. The response norm collapsed,
   so the plus/minus phases stopped giving useful contrastive information.

   `T = 0.5` kept the system mobile enough for the clamping perturbation to
   change output statistics while still allowing crisp class states to form.

4. **Use enough relaxation for the discrete Markov chain**

   The successful run used `1000` Metropolis proposals for the free phase and
   `1000` for each nudged phase. Earlier short probes could show promise, but
   they had much higher validation noise.

5. **Average over repeated initial states**

   Training used `Minit = 8`; evaluation used `64` repeats. This matters because
   a single discrete trajectory is a noisy draw from a multistable system. The
   learning signal should be based on output statistics, not one lucky or
   unlucky final state.

6. **Keep the Hamiltonian simple**

   No local potential, no double well, and no forced constant input. The terms
   were just interactions, trainable magnetic-field bias, and direct output
   clamping.

7. **Use the verified EP sign**

   Flipped-sign and target-free surrogate experiments were worse. The run that
   worked used the `plus - minus` Hamiltonian-derivative estimator documented
   in `Gradient.jl`.

8. **Use a stable optimizer configuration**

   Adam with `lr = 0.0015` and weight decay `1e-4` was stable for this run.
   These settings helped prevent parameter drift, but they were secondary to
   the discrete state set, finite temperature, and repeated-state averaging.

9. **Seed robustness is not solved**

   This is a real success case, not a proof that every seed works. At least one
   nearby discrete setup reached only a partial solution. The next engineering
   target is seed robustness, not sign debugging.

## What Did Not Make This Work

- It was not solved by the four-way one-hot input shortcut.
- It was not solved by continuous Langevin in the tested settings.
- It was not solved by setting temperature nearly to zero.
- It was not solved by a local potential or double well.
- It was not solved by the target-free surrogate gradient.
- It was not solved by flipping the EP gradient sign.

## Practical Baseline

For future 2-input XOR debugging, start from this run before changing multiple
things at once:

```text
2 -> 16 -> 2
discrete spins
Metropolis
T = 0.5
β = 1.0
free/nudged relaxation = 1000 / 1000
Minit = 8
eval repeats = 64
Adam lr = 0.0015
weight decay = 1e-4
weight seed = 13
```

Then test robustness by changing one of:

- training seed,
- weight seed,
- hidden size,
- temperature,
- relaxation budget,
- number of repeated initial states.

