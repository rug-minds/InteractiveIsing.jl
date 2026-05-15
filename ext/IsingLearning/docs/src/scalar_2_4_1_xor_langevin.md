# Scalar 2 -> 4 -> 1 XOR With LocalLangevin

This note documents the successful random-initialization `2 -> 4 -> 1` XOR
experiment and, just as importantly, the paths that did not work.

The goal was deliberately strict:

- two physical bipolar input spins;
- four hidden spins;
- one scalar bipolar output spin;
- no one-hot input trick;
- no hand-designed initialization as the actual learning result;
- no local potentials or double wells;
- learn with EqProp through the normal
  `Forward_and_Nudged -> contrastive_gradient -> Optimisers.update` path.

The relevant files are:

```text
ext/IsingLearning/experiments/simple_langevin_xor/analytic_2_4_1.jl
ext/IsingLearning/experiments/simple_langevin_xor/relaxation_sweep_2_4_1.jl
```

## What We Tried Before

### One-Hot Input XOR

The first statistical XOR success used a four-way one-hot input code. That was
useful for proving the learning machinery worked, but it was not an honest XOR
test: the four input cases were already separated by the encoding.

Conclusion: useful plumbing test, not a serious benchmark.

### Two-Input, Two-Output XOR

The honest two-input, two-output setup solved XOR with both discrete Metropolis
and tuned continuous Langevin. This used output targets:

```text
false -> [+1, -1]
true  -> [-1, +1]
```

This worked much more easily because the output layer has an internal
competition/contrast. The classifier only needs the correct output coordinate
to be larger.

Conclusion: good honest baseline, but easier than scalar output.

### Early Scalar 2 -> 4 -> 1 Runs

The scalar-output setup was much harder. Many early runs reached accuracy `1.0`
transiently but kept weak output margins. Typical MSE stalled around `0.5-0.7`.

This was misleading: accuracy could be correct while the scalar output was only
barely on the correct side of zero. The target was still `±1`, so a weak
correct-sign output was not a good solution.

Conclusion: scalar XOR must be judged by MSE/margin, not only accuracy.

### Stronger Clamping Alone

Increasing clamping strength alone did not solve the problem.

Observed behavior:

- `β = 1` was too weak;
- `β = 3` or `β = 5` was not better and often destabilized or weakened the
  useful response;
- `β = 2` was the useful range in the final recipe.

Conclusion: the bottleneck was not simply "make the teaching signal larger".
The sampler had to be in the right dynamical regime.

### More Relaxation Alone

More free/nudged sweeps were not monotonically better.

With `β = 2`, `T = 0.003`, `stepsize = 0.3`, `lr = 0.002`, and 1200 epochs:

| free/nudged | best observed behavior |
|---|---|
| `300/300` | reached accuracy `1.0`, MSE around `0.47` |
| `600/600` | lower MSE around `0.37`, but accuracy `0.75` |
| `1000/1000` | slower, MSE around `0.46` |
| `1500/1000` | best MSE in that sweep around `0.32` |
| `1000/1500` | MSE around `0.38` |

Conclusion: relaxation budget matters, but it interacts strongly with
temperature, step size, and learning rate. More sweeps by themselves are not the
answer.

## Analytic Capacity Check

Before continuing random training, we checked whether `2 -> 4 -> 1` can express
XOR at all.

It can.

Use four hidden units as corner detectors:

```math
h_{ab} = \operatorname{sign}(a x_1 + b x_2 - 1),
\qquad (a,b) \in \{(-1,-1),(-1,+1),(+1,-1),(+1,+1)\}.
```

Then read out:

```math
y = \operatorname{sign}(-h_{--} + h_{-+} + h_{+-} - h_{++}).
```

With this codebase's Hamiltonian convention,

```math
H = -\frac{1}{2}s^T J s - b^T s,
```

the analytic construction is:

- input-to-hidden weights: `A * (a, b)`;
- hidden bias: `-A`;
- hidden-to-output weights: `B * [-1, +1, +1, -1]`;
- output bias: `0`.

With `A = 2.0`, `B = 0.75`, `T = 0.001`, and `stepsize = 0.2`, the analytic
graph immediately evaluates at:

```text
accuracy = 1.0
MSE ≈ 0.016
output means ≈ [-0.85, +0.89, +0.87, -0.88]
```

Conclusion: the architecture is expressive enough. Failure from random init is
a training/basin/conditioning issue, not a capacity issue.

## What Finally Worked

The successful random-init regime was not the coldest or most exact relaxation.
It was a hotter, larger-step Langevin regime with moderate clamping.

Current best recipe:

```text
architecture  = 2 -> 4 -> 1
state         = continuous bounded [-1, +1]
dynamics      = LocalLangevin(adjusted=false)
T             = 0.005
stepsize      = 0.4
β             = 2.0
lr            = 0.002
weight_decay  = 0
Minit         = 8 or 16
free/nudged   = 300/300 or 600/600
```

Observed successful random-init runs:

```text
free/nudged = 600/600
Minit = 8
epoch 1500: MSE = 0.052574, accuracy = 1.0
```

```text
free/nudged = 300/300
Minit = 8
epoch 1600: MSE = 0.073809, accuracy = 1.0
```

```text
free/nudged = 600/600
Minit = 16
epoch 1500: MSE = 0.060262, accuracy = 1.0
```

These runs were from random initialization. They did not use the analytic
corner-detector weights as a starting point.

## What Made It Work

### 1. The Sampler Needed Enough Motion

Earlier runs were often too cold or too conservative. They could settle into
weak sign solutions, but the output margins stayed small.

The combination:

```text
T = 0.005
stepsize = 0.4
```

gave the continuous system enough motion to find useful basins and produce a
meaningful plus/minus response.

### 2. Clamping Had To Be Moderate

`β = 2` was better than both weaker and stronger clamping.

Interpretation: the scalar output needs a strong enough teaching perturbation,
but too much clamping drives a poor, nonlocal response rather than a useful
susceptibility.

### 3. Shorter Relaxation Was Sometimes Better

`300/300` and `600/600` worked better than blindly using very long phases.

Interpretation: for this small scalar task, the useful learning signal is not
necessarily the result of fully settling every phase. The system needs a
repeatable directional response, and overly long phases can move it into a
different basin or wash out the useful response.

### 4. MSE Revealed Progress That Accuracy Hid

Accuracy reached `1.0` in many mediocre runs. The meaningful improvement was
the scalar output moving toward `±1`.

Good runs had output means like:

```text
[-0.79, +0.77, +0.80, -0.73]
```

instead of weak correct signs like:

```text
[-0.20, +0.15, +0.40, -0.10]
```

### 5. The Main Remaining Problem Is Retention

The good runs often overshot after entering the good basin.

Example:

```text
epoch 1500: MSE ≈ 0.05, accuracy = 1.0
later:      MSE drifts upward
```

This means the learning rule can find the solution, but the current training
loop does not yet preserve it.

## What Did Not Make It Work

### Hand-Designed Initialization

The analytic initialization proves capacity, but it is not the result we want.
It is too specific and does not teach us much about general learning.

### Larger Hidden Layer Alone

The `2 -> 16 -> 2` setup works, but the scalar question was whether the minimal
`2 -> 4 -> 1` physical setup could learn. Since `2 -> 4 -> 1` now learns from
random init, increasing hidden units is not the fundamental answer.

### Strong Weight Decay

Small weight decay reduced some drift but capped solution quality. In the best
scalar runs, `weight_decay = 0` worked better.

### More Averaging Alone

Increasing `Minit` from 8 to 16 still reached a good MSE, but did not remove
overshoot. Averaging reduces noise; it does not replace learning-rate control.

## Current Interpretation

The scalar `2 -> 4 -> 1` problem is learnable with the current EqProp sign and
gradient plumbing.

The hard part is not expressivity. The hard part is the dynamical regime:

- too cold: the system freezes or gives weak response;
- too hot: output variance and drift dominate;
- too little clamping: weak scalar teaching signal;
- too much clamping: distorted response;
- too few useful updates: no basin discovery;
- too many updates after success: overshoot.

The current best explanation is:

```text
T/stepsize control basin exploration and response strength.
β controls the scalar teaching perturbation.
free/nudged sweeps control how much of that response is expressed.
lr controls whether the good basin is retained after discovery.
```

## Next Steps

The next practical improvement should target retention:

1. save and restore best parameters by validation MSE;
2. add learning-rate decay when MSE enters the good basin;
3. possibly freeze training once MSE is below a threshold;
4. log hidden-state means per XOR case to see whether the learned solution
   resembles corner detectors.

The problem is no longer "can it learn?". It can. The problem is making the
learning stable after it finds the scalar XOR solution.
