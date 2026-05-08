# XOR: Exact EP vs Short-Run Contrast

This note explains what the successful XOR run is doing mathematically, and why
it can learn even though the free phase is probably not at exact equilibrium
after 50 steps.

## The Objective We Evaluate

The successful XOR run does not train only a sign classifier.

For input case `x` and two-output target `y(x)`, the target is:

```text
XOR false: y = [ 1, -1]
XOR true:  y = [-1,  1]
```

Evaluation uses the mean relaxed output over repeated validation trajectories:

```math
\bar{o}_\theta(x)
= \frac{1}{R}\sum_{r=1}^R o_\theta^{(r)}(x).
```

The reported MSE is:

```math
L_{\mathrm{eval}}(\theta)
=
\frac{1}{|\mathcal{D}|}
\sum_{(x,y)\in\mathcal{D}}
\frac{1}{2}\left\|\bar{o}_\theta(x)-y\right\|^2.
```

Accuracy is only a secondary diagnostic:

```math
\hat{c}(x) = \arg\max_i \bar{o}_{\theta,i}(x).
```

So when the run reaches MSE below `0.1`, it is not merely learning the output
sign. The output means are moving toward the two-dimensional target vectors.

## Exact Equilibrium Propagation

In exact EP, the free state is an equilibrium of the energy:

```math
s_0^\star(\theta, x)
=
\arg\min_s H(\theta, s, x).
```

The task loss is added as a small clamping term:

```math
H_\beta(\theta, s, x, y)
= H(\theta, s, x) + \beta C(s,y),
```

with:

```math
C(s,y)=\frac{1}{2}\|o(s)-y\|^2.
```

The nudged equilibrium is:

```math
s_\beta^\star(\theta, x, y)
=
\arg\min_s H_\beta(\theta, s, x, y).
```

The one-sided EP estimator is based on the limit:

```math
\nabla_\theta L
\approx
\frac{\partial_\theta H(\theta, s_\beta^\star)
      - \partial_\theta H(\theta, s_0^\star)}
     {\beta},
\quad \beta \to 0.
```

The symmetric estimator used by the code is:

```math
\nabla_\theta L
\approx
\frac{
\partial_\theta H(\theta, s_{+\beta}^\star)
-
\partial_\theta H(\theta, s_{-\beta}^\star)}
{2\beta}.
```

This estimator assumes that the free and nudged states are equilibria.

## Why Exact Hard EP Can Fail Here

For a hard bounded linear Ising-like system, a unit's local energy often looks
approximately linear in that unit when all other units are fixed:

```math
H_i(s_i \mid s_{\setminus i})
= -h_i s_i + \mathrm{const}.
```

On a bounded interval `s_i ∈ [-1,1]`, the minimum is usually at a boundary:

```math
s_i^\star = \operatorname{sign}(h_i).
```

If the clamp is tiny, then the effective field changes from `h_i` to:

```math
h_i + \delta h_i.
```

If `h_i` is not near zero, `sign(h_i + δh_i) = sign(h_i)`, so the equilibrium
state does not move. If the sign crosses zero, the state jumps discontinuously.

That creates the old problem:

```text
small clamp + exact hard equilibrium
-> no response almost everywhere
-> discontinuous jumps near thresholds
-> poor infinitesimal EP signal
```

This is why the previous single-equilibrium hard-Ising approach was probably
the wrong first debugging target.

## What The Successful Run Actually Does

The successful run does not wait for exact equilibrium.

For each input `x`, target `y`, and random seed `r`, it runs:

```text
random/reset state
-> 50 free dynamics steps
-> save transient free state s_0^K
-> restore s_0^K
-> 50 plus-clamped steps under H + βC
-> save s_+^K
-> restore s_0^K
-> 50 minus-clamped steps under H - βC
-> save s_-^K
```

Here `K=50` for both free and nudged phases.

Mathematically, let `P_0^K` be the Markov transition operator for `K` free
steps, and `P_β^K` the transition operator for `K` clamped steps.

The free state distribution after `K` steps is:

```math
\mu_0^K(\theta, x)
=
\mu_{\mathrm{init}} P_0^K.
```

The plus and minus distributions are:

```math
\mu_+^K(\theta, x, y)
=
\mu_0^K P_{+\beta}^K,
```

```math
\mu_-^K(\theta, x, y)
=
\mu_0^K P_{-\beta}^K.
```

The code estimates:

```math
g_K(\theta)
=
\frac{1}{2\beta}
\left(
\mathbb{E}_{s\sim\mu_+^K}[\partial_\theta H(\theta,s)]
-
\mathbb{E}_{s\sim\mu_-^K}[\partial_\theta H(\theta,s)]
\right).
```

With finite samples, this becomes:

```math
\hat{g}_K(\theta)
=
\frac{1}{N}
\sum_{n=1}^{N}
\frac{
\partial_\theta H(\theta,s_{+,n}^K)
-
\partial_\theta H(\theta,s_{-,n}^K)}
{2\beta}.
```

That is exactly what the active code path does:

- `ForwardDynamics` obtains `s_0^K`.
- `NudgedDynamics.plus` starts from `s_0^K` and obtains `s_+^K`.
- `NudgedDynamics.minus` starts from the same `s_0^K` and obtains `s_-^K`.
- `contrastive_gradient` computes
  `∂θH(s_+^K) - ∂θH(s_-^K)`.
- `_collect_batch_gradient!` divides by `2β` and the number of sampled
  trajectories.

## Why This Can Learn Without Exact Equilibrium

For finite `K`, `g_K` is not the exact EP gradient of the equilibrium objective.
It is a gradient-like contrast for a different object: the behavior of a
finite-time stochastic relaxation process.

The clamp changes the transition operator:

```math
P_0^K
\quad\rightarrow\quad
P_\beta^K.
```

Even if the free state has not fully equilibrated, the next `K` steps can be
biased by the clamping term. The plus trajectory is nudged toward lower task
loss, while the minus trajectory is nudged away. Comparing their parameter
derivatives produces a direction that tends to make the plus trajectory more
energetically favorable than the minus trajectory.

In words:

```text
free state is a starting point
plus clamp says "from here, move in a target-helpful direction"
minus clamp says "from here, move in the opposite target direction"
the parameter update lowers energy of plus-like states relative to minus-like states
```

This can improve the finite-time output behavior even when the exact EP theorem
does not apply.

## What Objective Is This Closer To?

A useful way to think about the run is:

```math
L_K(\theta)
=
\mathbb{E}_{x,y}
\left[
\frac{1}{2}
\left\|
\mathbb{E}_{s\sim\mu_0^K(\theta,x)}[o(s)]
- y
\right\|^2
\right].
```

The implemented update is not an exact backpropagation gradient of `L_K`,
because it does not differentiate through every random transition step.

But it is a local contrastive estimator that asks:

```text
Do target-nudged short-run states get lower model energy than anti-nudged
short-run states?
```

When the answer is increasingly yes, the free finite-time output distribution
can move toward the target distribution. That is what the successful XOR run
showed empirically.

## Why 50 Steps Can Work

The old 1500-step discussion was about trying to get close to a deterministic
or near-deterministic equilibrium before applying a small perturbation.

The successful run does something else:

- it uses simple one-hot all-to-all input/output;
- it uses direct two-spin output clamping;
- it samples multiple starts;
- it keeps a cold but finite temperature;
- it uses finite `β=0.05`;
- it only needs a useful finite-time contrast, not exact convergence.

So the 50-step number should be interpreted as:

```text
enough steps for a useful short-run contrast in this simplified stochastic task
```

not:

```text
enough steps to prove the system reached equilibrium
```

## What This Means For Claims

The successful XOR run supports this claim:

```text
The current IsingLearning process path can learn XOR using a repeated-state
finite-time contrastive EP-like update.
```

It does not yet support this stronger claim:

```text
The hard bounded Ising system is performing exact equilibrium propagation at
true equilibria.
```

For the stronger claim, we would need to show that free and nudged phases have
actually equilibrated, then verify that the learned update remains useful in
the small-β limit.
