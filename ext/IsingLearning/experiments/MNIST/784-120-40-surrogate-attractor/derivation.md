# Surrogate MMC Attractor Learning for Ising-Machine Classification

## 1. Goal

We want a learning rule for an Ising-like classifier where classification is assumed to happen through the local minimum / attractor reached by the dynamics.

For input `x`, the machine defines an energy

```math
E_\theta(\sigma; x),
```

with binary spins

```math
\sigma_i \in \{-1,+1\}.
```

A run of the machine starts from some initialization or noisy dynamics and eventually reaches a local minimum

```math
m \in \mathcal M_\theta(x).
```

The output/classification is read from the final minimum `m`.

The true objective would ideally improve the probability that the machine reaches minima that classify correctly:

```math
J_{\rm actual}(\theta)
=
\mathbb E_{m \sim p_{\rm actual}(m|x;\theta)}[R_y(m)],
```

where:

- `p_actual(m | x; θ)` is the real hitting distribution over local minima produced by the machine dynamics,
- `R_y(m)` is a reward measuring how well minimum `m` classifies target label `y`.

The problem is that `p_actual` is generally unknown. It depends on basin volumes, barriers, update dynamics, noise, stopping rule, and initialization distribution.

The proposal here is to use a **surrogate MMC/Boltzmann ranking rule**: sample minima using the real machine, but update parameters **as if** the sampled minima came from a Boltzmann-weighted minimum ensemble.

This gives a simple REINFORCE-like attractor learning rule.

---

## 2. Surrogate distribution over minima

Instead of using the unknown true hitting distribution, define a surrogate distribution over local minima:

```math
\tilde p_\theta(m|x)
=
\frac{\exp[-E_\theta(m;x)/T_s]}
{\sum_{m' \in \mathcal M_\theta(x)} \exp[-E_\theta(m';x)/T_s]}.
```

Here `T_s` is a **surrogate temperature**, not necessarily the physical temperature of the machine.

This says:

```math
\text{lower-energy minima are treated as more probable.}
```

The corresponding surrogate objective is

```math
\tilde J(\theta)
=
\mathbb E_{m \sim \tilde p_\theta}[R_y(m)].
```

In practice, we do **not** sample exactly from `p̃`. We sample minima from the actual machine dynamics and use the Boltzmann-minimum score as a heuristic credit signal.

So the learning rule is biased relative to the true hitting-distribution objective, but it may approximate it if energy depth correlates with reachability, basin size, or stability.

---

## 3. Score-function derivation

For the surrogate distribution

```math
\tilde p_\theta(m)
\propto
\exp[-E_\theta(m)/T_s],
```

we have

```math
\log \tilde p_\theta(m)
=
-\frac{1}{T_s}E_\theta(m)-\log \tilde Z_\theta.
```

Therefore

```math
\nabla_\theta \log \tilde p_\theta(m)
=
-\frac{1}{T_s}
\left[
\nabla_\theta E_\theta(m)
-
\mathbb E_{\tilde p}[\nabla_\theta E_\theta]
\right].
```

The score-function gradient of the surrogate expected reward is

```math
\nabla_\theta \tilde J
=
\mathbb E_{\tilde p}
\left[
R_y(m)\nabla_\theta\log \tilde p_\theta(m)
\right].
```

Substituting the score gives

```math
\nabla_\theta \tilde J
=
-\frac{1}{T_s}
\operatorname{Cov}_{\tilde p}
\left(
R_y(m),
\nabla_\theta E_\theta(m)
\right).
```

Equivalently, with a baseline `b`,

```math
\nabla_\theta \tilde J
=
-\frac{1}{T_s}
\mathbb E_{\tilde p}
\left[
(R_y(m)-b)\nabla_\theta E_\theta(m)
\right],
```

where an ideal baseline is

```math
b = \mathbb E_{\tilde p}[R_y(m)].
```

In practice, `b` can be a batch mean, exponential moving average, or class-dependent baseline.

---

## 4. Ising energy and parameter update

Consider the Ising energy

```math
E_\theta(\sigma;x)
=
-\sum_{i<j}J_{ij}\sigma_i\sigma_j
-\sum_i h_i\sigma_i
+ E_{\rm input}(\sigma;x).
```

For pair couplings,

```math
\frac{\partial E}{\partial J_{ij}}
=
-\sigma_i\sigma_j.
```

For biases,

```math
\frac{\partial E}{\partial h_i}
=
-\sigma_i.
```

Reward ascent on the surrogate objective gives

```math
\Delta J_{ij}
\propto
\frac{\eta}{T_s}(R_y(m)-b)m_i m_j,
```

and

```math
\Delta h_i
\propto
\frac{\eta}{T_s}(R_y(m)-b)m_i.
```

So the rule is:

```math
\boxed{
\Delta J_{ij}
=
\alpha (R_y(m)-b)m_i m_j
}
```

```math
\boxed{
\Delta h_i
=
\alpha (R_y(m)-b)m_i
}
```

with

```math
\alpha = \eta/T_s.
```

Interpretation:

- if a sampled minimum has above-baseline reward, lower its energy;
- if a sampled minimum has below-baseline reward, raise its energy;
- the update is local in the sampled spin correlations.

For a good minimum, `R > b`, the update reinforces the spin pattern `m` as an attractor-like state.

For a bad minimum, `R < b`, the update anti-reinforces it.

---

## 5. Batch version

For a batch of sampled minima

```math
\{m^{(a)}\}_{a=1}^B,
```

with rewards

```math
R^{(a)} = R_y(m^{(a)}),
```

choose

```math
b = \frac{1}{B}\sum_{a=1}^B R^{(a)}.
```

Then update

```math
\Delta J_{ij}
=
\alpha
\frac{1}{B}
\sum_{a=1}^B
(R^{(a)}-b)m_i^{(a)}m_j^{(a)}.
```

Equivalently,

```math
\Delta J_{ij}
=
\alpha
\operatorname{Cov}_{\rm samples}
\left(
R,
 m_i m_j
\right).
```

Similarly,

```math
\Delta h_i
=
\alpha
\operatorname{Cov}_{\rm samples}(R,m_i).
```

This covariance form is useful because it automatically removes a global reward offset.

---

## 6. Classification reward choices

Suppose the output is read from output spins or logits `o(m)`.

Possible reward definitions:

### 6.1 Negative loss reward

Use cross-entropy loss

```math
\ell_y(m) = -\log p_y(m),
```

and reward

```math
R_y(m) = -\ell_y(m).
```

Then above-baseline minima are those with lower-than-average classification loss.

### 6.2 Correct/incorrect reward

Use

```math
R_y(m)=
\begin{cases}
+1, & \text{if } m \text{ classifies correctly},\\
0, & \text{otherwise}.
\end{cases}
```

or

```math
R_y(m)=
\begin{cases}
+1, & \text{correct},\\
-1, & \text{incorrect}.
\end{cases}
```

This is simple but high-variance.

### 6.3 Margin reward

If the output has class scores `u_k(m)`, define

```math
R_y(m)
=
u_y(m)-\log\sum_k e^{u_k(m)}.
```

This is the log-probability of the correct class. It gives a smoother reward than hard correctness.

### 6.4 Energy-margin reward

If each class has an associated output energy or class readout energy, use a margin:

```math
R_y(m)
=
-\left[E_{\rm out}(y,m)-\min_{k\neq y}E_{\rm out}(k,m)\right].
```

This rewards minima where the correct class has lower output energy than competing classes.

---

## 7. Algorithm

For each training example `(x, y)`:

1. Clamp or encode the input `x` into the machine.
2. Run the actual machine dynamics from one or more initializations.
3. Let each run settle to a local minimum `m`.
4. Read out the classification from `m`.
5. Compute reward `R_y(m)`.
6. Compute a baseline `b`, for example the batch mean reward.
7. Update parameters using

```math
\Delta J_{ij}
=
\alpha (R_y(m)-b)m_i m_j,
```

```math
\Delta h_i
=
\alpha (R_y(m)-b)m_i.
```

For a batch of minima, average the updates.

Pseudo-code:

```text
for each training step:
    samples = []

    for a in 1:B:
        initialize state σ
        clamp input x
        run dynamics until local minimum m_a
        compute reward R_a = R_y(m_a)
        store (m_a, R_a)

    b = mean(R_a over samples)

    ΔJ = 0
    Δh = 0

    for (m_a, R_a) in samples:
        advantage = R_a - b
        ΔJ += advantage * outer(m_a, m_a)
        Δh += advantage * m_a

    J += α * ΔJ / B
    h += α * Δh / B
```

For constrained topologies, only update allowed couplings.

For symmetric Ising couplings, symmetrize the update:

```math
\Delta J_{ij}=\Delta J_{ji}.
```

Usually set

```math
J_{ii}=0.
```

---

## 8. Relation to EqProp

Classical deterministic EqProp uses two nearby equilibria:

```math
m^0 = \arg\min_s E_\theta(s),
```

and

```math
m^\beta = \arg\min_s [E_\theta(s)+\beta C(s)].
```

The gradient is estimated from a contrast between the free and nudged equilibria.

The surrogate MMC attractor rule does something different:

- it samples one or more free minima;
- it does **not** require a nudged phase;
- it assigns each minimum a reward;
- it reinforces or anti-reinforces that minimum as if minima were Boltzmann-weighted by energy.

So this is closer to a REINFORCE / policy-gradient rule over attractors than to classical EqProp.

A compact comparison:

```math
\text{EqProp:}
\quad
\Delta \theta
\sim
\frac{1}{\beta}
\left[
\partial_\theta F_\beta(m^\beta)-\partial_\theta F_0(m^0)
\right]
```

```math
\text{surrogate attractor REINFORCE:}
\quad
\Delta \theta
\sim
-(R-b)\partial_\theta E(m).
```

For Ising couplings:

```math
\text{surrogate attractor REINFORCE:}
\quad
\Delta J_{ij}
\sim
(R-b)m_i m_j.
```

---

## 9. What is biased about the rule?

The true dynamics samples minima from

```math
p_{\rm actual}(m|x;\theta).
```

The surrogate rule acts as if they were sampled from

```math
\tilde p_\theta(m|x)
\propto
\exp[-E_\theta(m;x)/T_s].
```

The expected update is therefore not generally the exact gradient of

```math
J_{\rm actual}(\theta)
=
\mathbb E_{p_{\rm actual}}[R(m)].
```

It is a heuristic gradient-like direction based on the assumption that making good minima lower in energy makes them more likely to be reached.

This assumption is plausible when lower-energy minima tend to be deeper, more stable, or have larger basins. But it can fail: a low-energy minimum can have a tiny basin, and a higher-energy minimum can have a huge basin.

So the rule is best understood as:

```math
\boxed{
\text{energy-ranking credit assignment over sampled attractors}
}
```

not as an exact gradient of the real hitting distribution.

---

## 10. Possible improvements

### 10.1 Add stability weighting

A local minimum is characterized by positive one-spin flip margins

```math
\Delta_i(m)
=
E(\operatorname{flip}_i m)-E(m)>0.
```

A stability score could be

```math
S(m)=\sum_i \Delta_i(m),
```

or

```math
S_{\min}(m)=\min_i \Delta_i(m).
```

One can modify the reward:

```math
R'(m)=R(m)+\lambda S(m),
```

or use stability only for correct minima.

This encourages not merely low-energy minima, but robust minima.

### 10.2 Penalize wrong attractors more directly

For wrong minima, use a stronger negative reward:

```math
R_y(m)=
\begin{cases}
+1, & \text{correct},\\
-\gamma, & \text{wrong},
\end{cases}
```

with `γ > 0`.

Larger `γ` suppresses wrong attractors more aggressively.

### 10.3 Use multiple initializations per input

Because the relevant distribution is over reachable attractors, use several initializations/noise seeds per input.

This lets the update see both good and bad basins for the same input.

### 10.4 Use a moving baseline

Instead of the current batch mean, use

```math
b \leftarrow (1-\rho)b + \rho R.
```

This reduces variance and prevents very small batches from producing unstable advantage estimates.

### 10.5 Combine with a local readout gradient

If the output layer is differentiable, train output/readout parameters with ordinary supervised gradients, while training recurrent/Ising couplings with the attractor rule.

---

## 11. Suggested experiment

A minimal test:

1. Use a small Ising machine with input spins, hidden spins, and output spins.
2. Clamp input spins.
3. Randomly initialize hidden/output spins.
4. Run deterministic or noisy relaxation until a local minimum.
5. Decode class from output spins.
6. Assign reward:

```math
R=1 \quad \text{if correct},
```

```math
R=0 \quad \text{if wrong}.
```

7. Use batch baseline:

```math
b=\text{mean reward in batch}.
```

8. Update recurrent couplings:

```math
\Delta J_{ij}
=\alpha (R-b)m_i m_j.
```

9. Track:

- classification accuracy,
- average reward,
- number of distinct minima reached per input,
- energy of correct vs wrong minima,
- basin frequencies from repeated random initializations.

A useful diagnostic is whether correct minima become both lower in energy **and** more frequently reached. If energy decreases but hit probability does not improve, the rule is optimizing the surrogate but not reshaping the relevant basins enough.

---

## 12. Main takeaway

The proposed rule is:

```math
\boxed{
\Delta J_{ij}
=\alpha (R_y(m)-b)m_i m_j
}
```

for minima `m` sampled from the actual machine dynamics.

It can be interpreted as a REINFORCE-like rule using the surrogate Boltzmann-minimum score

```math
\tilde p_\theta(m)
\propto
\exp[-E_\theta(m)/T_s].
```

It is not an exact gradient of the true attractor hitting distribution unless the true hitting distribution matches the surrogate. But it is a simple, local, testable attractor-ranking rule:

```math
\text{make good sampled minima lower in energy; make bad sampled minima higher in energy.}
```
