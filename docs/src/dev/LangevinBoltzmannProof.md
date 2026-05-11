# Langevin Boltzmann Proof

This is a scratch proof for the exact Langevin algorithms implemented in
`src/MCAlgorithms/Algorithms/Langevin`.

The goal is not to prove a generic Langevin method. The goal is to check the
actual transition kernels used by this code.

Implementation note: `GlobalLangevin` and `BlockLangevin` now expose one state
write per `Processes.step!`, but `adjusted=true` still accepts/rejects at the
proposal scope: all active spins for global and the selected block for block
Langevin. Accepted vector proposals are then streamed into the graph one
`FlipProposal` at a time. Sections below that discuss historical immediate
`MultiSpinProposal` writes are kept as scratch notes until this proof is fully
rewritten around the current streamed-write implementation.

## Target Distribution

For a graph state `x`, the intended finite-temperature Boltzmann target is

```math
\pi(x) = Z^{-1}\exp(-H(x)/T),
```

on the valid state domain defined by the model's layer state bounds. Outside
that domain, the target density is treated as zero.

Code representation:

- `T` is read from the model with `temp(model)` in
  `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:173`,
  `src/MCAlgorithms/Algorithms/Langevin/GlobalLangevin.jl:91`, and
  `src/MCAlgorithms/Algorithms/Langevin/BlockLangevin.jl:100`.
- The energy `H(x)` itself is not directly evaluated in the adjusted kernels.
  The code uses `ΔE = H(y) - H(x)` through `calculate(ΔH(), ...)`, e.g.
  `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:234`,
  `src/MCAlgorithms/Algorithms/Langevin/GlobalLangevin.jl:177`, and
  `src/MCAlgorithms/Algorithms/Langevin/BlockLangevin.jl:178`.
- The valid support is represented by the layer state bounds
  `low_state = local_states[1]` and `high_state = local_states[end]`, e.g.
  `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:205-208`,
  `src/MCAlgorithms/Algorithms/Langevin/GlobalLangevin.jl:131-134`, and
  `src/MCAlgorithms/Algorithms/Langevin/BlockLangevin.jl:134-137`.
- In adjusted mode, proposals outside that support are rejected with
  `_in_bounds`, e.g. `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:229-230`,
  `src/MCAlgorithms/Algorithms/Langevin/GlobalLangevin.jl:147-162`, and
  `src/MCAlgorithms/Algorithms/Langevin/BlockLangevin.jl:150-163`.

The proof below assumes:

- `T = temp(model) > 0`.
- `stepsize[]` is fixed during the sampling phase.
- `calculate(ΔH(), hamiltonian, model, proposal)` returns exactly
  `H(y) - H(x)`.
- `calculate(d_iH(), hamiltonian, model, i)` returns the derivative component
  used by the proposal.
- `_finite_derivative` does not change the derivative, meaning derivatives are
  finite on the states being sampled.
- Hamiltonian cache updates preserve consistency with the graph state after an
  accepted proposal.

If these assumptions do not hold, the proof below does not establish Boltzmann
stationarity for the physical Hamiltonian.

## Common MALA Algebra

The adjusted Langevin paths in the code use proposals of the form

```math
y = x - \eta \nabla H(x) + \sqrt{2\eta T}\,\xi,
\qquad \xi \sim \mathcal{N}(0, I).
```

Code representation:

- `η` is `max(stepsize[], epsT)`:
  `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:175`,
  `src/MCAlgorithms/Algorithms/Langevin/GlobalLangevin.jl:93`, and
  `src/MCAlgorithms/Algorithms/Langevin/BlockLangevin.jl:102`.
- `σ = sqrt(2ηT)` is computed as
  `sqrt(SType(2) * η * t)`:
  `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:181`,
  `src/MCAlgorithms/Algorithms/Langevin/GlobalLangevin.jl:106`, and
  `src/MCAlgorithms/Algorithms/Langevin/BlockLangevin.jl:103`.
- The derivative components are computed with
  `calculate(d_iH(), hamiltonian, model, spin_idx)`:
  `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:201`,
  `src/MCAlgorithms/Algorithms/Langevin/GlobalLangevin.jl:124`, and
  `src/MCAlgorithms/Algorithms/Langevin/BlockLangevin.jl:127`.
- The adjusted drift is the raw product `η * derivative`; the cap is only used
  when `use_adjusted` is false:
  `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:211-212`,
  `src/MCAlgorithms/Algorithms/Langevin/GlobalLangevin.jl:136`, and
  `src/MCAlgorithms/Algorithms/Langevin/BlockLangevin.jl:139`.
- The normal noise draw is `randn(rng, SType)`, and the proposed state is
  `old_state - drift_step + σ * noise` locally or the analogous vector/block
  expression globally:
  `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:214-216`,
  `src/MCAlgorithms/Algorithms/Langevin/GlobalLangevin.jl:138-139`, and
  `src/MCAlgorithms/Algorithms/Langevin/BlockLangevin.jl:141-142`.

The corresponding proposal density is a Gaussian with mean

```math
\mu(x) = x - \eta \nabla H(x)
```

Code representation:

- Local forward mean: `forward_mean = old_state - drift_step` in
  `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:239`.
- Global forward mean: `forward_mean = old_state - drift_step` in
  `src/MCAlgorithms/Algorithms/Langevin/GlobalLangevin.jl:149`.
- Block forward mean is passed inline as `old_state - drift_step` in
  `src/MCAlgorithms/Algorithms/Langevin/BlockLangevin.jl:152`.

and covariance

```math
2\eta T\,I.
```

Code representation:

- The code stores the scalar denominator `four_ηT = 4ηT` as
  `SType(4) * η * max(t, epsT)` in
  `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:191`,
  `src/MCAlgorithms/Algorithms/Langevin/GlobalLangevin.jl:113`, and
  `src/MCAlgorithms/Algorithms/Langevin/BlockLangevin.jl:118`.
- Since `_mala_log_kernel` divides squared displacement by `four_ηT`, this is
  the Gaussian log-density for covariance `2ηT I`, up to constants.

Therefore, up to additive constants that cancel in a proposal-density ratio,

```math
\log q(y \mid x)
=
-\frac{\|y - \mu(x)\|^2}{4\eta T}.
```

This is exactly what `_mala_log_kernel(x, mean, four_ηT)` computes:

```julia
dx = x - mean
return -(dx * dx) / four_ηT
```

Code representation:

- `_mala_log_kernel` is defined in
  `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:74-77`.
- Local forward/reverse calls are in
  `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:248-249`.
- Global forward/reverse sums are in
  `src/MCAlgorithms/Algorithms/Langevin/GlobalLangevin.jl:150` and
  `src/MCAlgorithms/Algorithms/Langevin/GlobalLangevin.jl:192`.
- Block forward/reverse sums are in
  `src/MCAlgorithms/Algorithms/Langevin/BlockLangevin.jl:152` and
  `src/MCAlgorithms/Algorithms/Langevin/BlockLangevin.jl:193`.

For multi-spin proposals, the code sums this scalar expression over all moved
coordinates, which is the log-density of the diagonal Gaussian proposal.

Metropolis-Hastings accepts `x -> y` with

```math
\alpha(x,y)
=
\min\left(
    1,
    \frac{\pi(y)q(x\mid y)}{\pi(x)q(y\mid x)}
\right).
```

With the Boltzmann target,

```math
\frac{\pi(y)}{\pi(x)}
=
\exp(-(H(y)-H(x))/T).
```

So the log acceptance ratio is

```math
\log r(x,y)
=
-\frac{H(y)-H(x)}{T}
+ \log q(x\mid y)
- \log q(y\mid x).
```

The code implements this as

```julia
log_acceptance = -ΔE / t + log_reverse_q - log_forward_q
```

Code representation:

- Local: `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:250`.
- Global: `src/MCAlgorithms/Algorithms/Langevin/GlobalLangevin.jl:199`.
- Block: `src/MCAlgorithms/Algorithms/Langevin/BlockLangevin.jl:200`.

and accepts when

```julia
log_acceptance >= 0 || log(rand(rng, SType)) < log_acceptance
```

Code representation:

- Local: `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:251`.
- Global: `src/MCAlgorithms/Algorithms/Langevin/GlobalLangevin.jl:200`.
- Block: `src/MCAlgorithms/Algorithms/Langevin/BlockLangevin.jl:201`.

which is equivalent to accepting with probability `min(1, exp(log_acceptance))`.

For states `x != y` in the valid domain, the transition density is

```math
P(x,dy) = q(y\mid x)\alpha(x,y)\,dy.
```

Then

```math
\pi(x)q(y\mid x)\alpha(x,y)
=
\min(\pi(x)q(y\mid x), \pi(y)q(x\mid y))
=
\pi(y)q(x\mid y)\alpha(y,x).
```

This is detailed balance. Therefore the Boltzmann distribution is stationary
for the adjusted kernel.

Rejected moves contribute the remaining probability to staying at `x`, which is
the usual Metropolis-Hastings self-transition and is required for the Markov
kernel to conserve probability.

## `LocalLangevin(adjusted=true)`

The local code moves one coordinate per `Processes.step!`.

For selected spin `i`, with all other coordinates fixed, the code proposes

```math
y_i = x_i - \eta\,\partial_i H(x) + \sqrt{2\eta T}\,\xi.
```

Code representation:

- The selected spin index is `spin_idx = active_spins[k]` in
  `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:195-200`.
- The derivative `∂ᵢH(x)` is `derivative = calculate(dh, hamiltonian, model, spin_idx)`
  in `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:201`.
- `old_state` is `x_i`, `trial_state` is `y_i`, and the proposal formula is
  implemented in `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:214-216`.

The forward log proposal density used by the code is

```math
\log q_i(y_i\mid x)
=
-\frac{(y_i - (x_i-\eta\partial_i H(x)))^2}{4\eta T}.
```

Code representation:

- `forward_mean = old_state - drift_step` in
  `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:239`.
- `log_forward_q = _mala_log_kernel(new_state, forward_mean, four_ηT)` in
  `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:248`.

To compute the reverse density, the code temporarily writes `y_i` into the graph
state, recalculates

```math
\partial_i H(y),
```

Code representation:

- The code temporarily writes `spins[spin_idx] = new_state` in
  `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:241`.
- It recomputes the derivative at the proposed state in
  `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:242-243`.
- It restores the old state in
  `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:246`.

then computes

```math
\log q_i(x_i\mid y)
=
-\frac{(x_i - (y_i-\eta\partial_i H(y)))^2}{4\eta T}.
```

Code representation:

- `reverse_mean = new_state - reverse_drift_step` in
  `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:245`.
- `log_reverse_q = _mala_log_kernel(old_state, reverse_mean, four_ηT)` in
  `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:249`.

The code computes `ΔE` from the single-spin `FlipProposal`, so under the
assumption that this is exact,

```math
\Delta E = H(y) - H(x).
```

Code representation:

- The one-spin proposal is represented by `FlipProposal` fields
  `at_idx`, `from_val`, and `to_val` in
  `src/Proposals/FlipProposal.jl:6-12`.
- Local Langevin constructs the trial proposal in
  `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:233`.
- The code computes `ΔE` in
  `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:234`.

Therefore the local adjusted acceptance rule is exactly the
Metropolis-Hastings rule for the one-coordinate Langevin proposal.

The selected coordinate is controlled by an internal random-cyclic sweep cursor.
For a fixed selected coordinate, the adjusted coordinate kernel leaves
`π` invariant by detailed balance. A deterministic composition of invariant
single-coordinate kernels also leaves `π` invariant, and a random mixture over
cyclic offsets also leaves `π` invariant. Therefore the local adjusted algorithm
has `π` stationary.

Code representation:

- The cursor state is initialized as `sweep_position`, `sweep_offset`, and
  `group_remaining` in
  `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:119-121`.
- A new cycle chooses `sweep_offset = rand(rng, 0:(n - 1))` in
  `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:146`.
- Each call advances the cursor in
  `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:150-164` and calls it
  at `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:266`.

Out-of-bounds local proposals are rejected. This is consistent with a target
distribution whose support is only the valid interval for that spin.

## `GlobalLangevin(adjusted=true)`

The global code moves all active spins in one `MultiSpinProposal`.

For active index set `A`, the code computes each derivative at the original
state `x`:

```math
d_j = \partial_j H(x), \qquad j \in A.
```

Code representation:

- The active set is `active_spins = _active_spin_vector(model)` in
  `src/MCAlgorithms/Algorithms/Langevin/GlobalLangevin.jl:98`.
- Each derivative is computed in
  `src/MCAlgorithms/Algorithms/Langevin/GlobalLangevin.jl:123-127`.

It proposes

```math
y_j = x_j - \eta d_j + \sqrt{2\eta T}\xi_j,
\qquad j \in A,
```

with independent standard normal noise per coordinate. The forward log density
is accumulated as

Code representation:

- `old_state = spins[spin_idx]` is `x_j` in
  `src/MCAlgorithms/Algorithms/Langevin/GlobalLangevin.jl:138`.
- `trial_state = old_state - drift_step + randn(...) * σ` is `y_j` in
  `src/MCAlgorithms/Algorithms/Langevin/GlobalLangevin.jl:139`.
- The loop calls `randn(rng, SType)` separately for each active coordinate, so
  the Gaussian proposal is diagonal/independent in
  `src/MCAlgorithms/Algorithms/Langevin/GlobalLangevin.jl:123-152`.

```math
\log q_A(y_A\mid x)
=
-\sum_{j\in A}
\frac{(y_j - (x_j-\eta\partial_j H(x)))^2}{4\eta T}.
```

Code representation:

- The forward mean is `forward_mean = old_state - drift_step` in
  `src/MCAlgorithms/Algorithms/Langevin/GlobalLangevin.jl:149`.
- The sum is accumulated by
  `log_forward_q += _mala_log_kernel(new_vals[pos], forward_mean, four_ηT)` in
  `src/MCAlgorithms/Algorithms/Langevin/GlobalLangevin.jl:150`.

If any coordinate is outside its bounds, the whole proposal is rejected. For
in-bounds proposals, the code constructs a `MultiSpinProposal` and computes
`ΔE`. The generic fallback for multi-spin `ΔH` applies the represented
single-spin changes sequentially, restores the original graph state, and
returns the accumulated energy change. Specialized Hamiltonian methods may
replace this, but the proof requires the returned value to equal

```math
H(y)-H(x).
```

Code representation:

- The multi-spin proposal storage is `MultiSpinProposal` in
  `src/Proposals/FlipProposal.jl:83-95`.
- Global Langevin constructs `proposal_trial` in
  `src/MCAlgorithms/Algorithms/Langevin/GlobalLangevin.jl:165`.
- Global Langevin computes `ΔE` in
  `src/MCAlgorithms/Algorithms/Langevin/GlobalLangevin.jl:177`.
- The generic multi-spin fallback accumulates sequential single-spin energy
  changes and restores the original state in
  `src/MCAlgorithms/Hamiltonians/Functionals.jl:38-52` and
  `src/MCAlgorithms/Hamiltonians/Functionals.jl:64-78`.

For the reverse density, the code temporarily writes the whole proposed vector
`y_A` into the graph state, recomputes each active derivative at `y`, and sums

```math
\log q_A(x_A\mid y)
=
-\sum_{j\in A}
\frac{(x_j - (y_j-\eta\partial_j H(y)))^2}{4\eta T}.
```

Code representation:

- The code writes the proposed global state in
  `src/MCAlgorithms/Algorithms/Langevin/GlobalLangevin.jl:181-183`.
- It recomputes reverse derivatives at `y` in
  `src/MCAlgorithms/Algorithms/Langevin/GlobalLangevin.jl:185-190`.
- It accumulates the reverse log density in
  `src/MCAlgorithms/Algorithms/Langevin/GlobalLangevin.jl:191-192`.
- It restores the original state in
  `src/MCAlgorithms/Algorithms/Langevin/GlobalLangevin.jl:195-197`.

It then restores `x_A`, computes

```math
-\Delta E/T + \log q_A(x_A\mid y) - \log q_A(y_A\mid x),
```

Code representation:

- `log_acceptance = -ΔE / t + log_reverse_q - log_forward_q` in
  `src/MCAlgorithms/Algorithms/Langevin/GlobalLangevin.jl:199`.

and accepts with the corresponding Metropolis-Hastings probability.

Thus, under the assumptions above, `GlobalLangevin(adjusted=true)` satisfies
detailed balance with the Boltzmann distribution on the bounded active state
domain.

`group_steps` repeats this same invariant kernel multiple times inside one
`Processes.step!`. A composition of kernels that each leave `π` invariant also
leaves `π` invariant.

## `BlockLangevin(adjusted=true)`

`BlockLangevin` is the same proof as `GlobalLangevin`, restricted to a block
`B` drawn from the shuffled active-index order.

For a fixed block `B`, the proposal density is the diagonal Gaussian over the
coordinates in `B`, with mean

```math
x_B - \eta \nabla_B H(x).
```

Code representation:

- The block is selected by `_fill_langevin_block!` in
  `src/MCAlgorithms/Algorithms/Langevin/BlockLangevin.jl:78-89`, called from
  `src/MCAlgorithms/Algorithms/Langevin/BlockLangevin.jl:122`.
- Each block derivative is computed in
  `src/MCAlgorithms/Algorithms/Langevin/BlockLangevin.jl:126-130`.
- Each block trial coordinate is computed in
  `src/MCAlgorithms/Algorithms/Langevin/BlockLangevin.jl:141-143`.

The code computes the forward log density at `x`, writes the proposed block
state to compute the reverse derivatives at `y`, computes the reverse log
density, and uses

```math
-\Delta E/T + \log q_B(x_B\mid y) - \log q_B(y_B\mid x).
```

Code representation:

- Block forward log density is accumulated in
  `src/MCAlgorithms/Algorithms/Langevin/BlockLangevin.jl:150-153`.
- `proposal_trial` and `ΔE` are computed in
  `src/MCAlgorithms/Algorithms/Langevin/BlockLangevin.jl:166` and
  `src/MCAlgorithms/Algorithms/Langevin/BlockLangevin.jl:178`.
- The proposed block is written in
  `src/MCAlgorithms/Algorithms/Langevin/BlockLangevin.jl:182-184`.
- Reverse derivatives and reverse log density are computed in
  `src/MCAlgorithms/Algorithms/Langevin/BlockLangevin.jl:186-193`.
- The old block is restored in
  `src/MCAlgorithms/Algorithms/Langevin/BlockLangevin.jl:196-198`.
- The log acceptance ratio is computed in
  `src/MCAlgorithms/Algorithms/Langevin/BlockLangevin.jl:200`.

So the fixed-block kernel satisfies detailed balance under the same
assumptions. The block selection is independent of the current state except
through the current active index list. A random mixture over such invariant
block kernels also leaves `π` invariant, assuming the active index list itself
is fixed during sampling.

`group_steps` again composes invariant kernels.

## What Is Not Proved Correct

The unadjusted paths are not proved Boltzmann-correct here.

In the unadjusted code:

- the drift is capped with `_langevin_drift_step`;
- out-of-bounds proposals are reflected with `_reflect_to_bounds`;
- the proposal is always accepted;
- no reverse/forward proposal-density ratio is used;
- no `exp(-ΔH/T)` factor is used.

Code representation:

- Drift capping is `_langevin_drift_step`, defined in
  `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:70-72`, and selected
  only when `use_adjusted` is false, e.g.
  `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:212`,
  `src/MCAlgorithms/Algorithms/Langevin/GlobalLangevin.jl:136`, and
  `src/MCAlgorithms/Algorithms/Langevin/BlockLangevin.jl:139`.
- Reflection is `_reflect_to_bounds`, defined in
  `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:49-64`, and used in
  the unadjusted paths at
  `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:221`,
  `src/MCAlgorithms/Algorithms/Langevin/GlobalLangevin.jl:142`, and
  `src/MCAlgorithms/Algorithms/Langevin/BlockLangevin.jl:143`.
- The unadjusted local path constructs an accepted proposal and writes the new
  state directly in `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:223-228`.
- The unadjusted global path constructs an accepted `MultiSpinProposal` and
  writes all new states in
  `src/MCAlgorithms/Algorithms/Langevin/GlobalLangevin.jl:167-174`.
- The unadjusted block path does the same for the block in
  `src/MCAlgorithms/Algorithms/Langevin/BlockLangevin.jl:168-175`.

That is not the Metropolis-Hastings kernel above. It may be useful as an
interactive dynamics or relaxation method, but this code does not establish
that it samples the Boltzmann distribution exactly.

The `T <= 0` adjusted branch is also not finite-temperature Boltzmann sampling.
The code accepts only finite non-increasing energy moves:

```julia
accept_move = isfinite(ΔE) && ΔE <= zero(SType)
```

Code representation:

- Local: `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:236-237`.
- Global: `src/MCAlgorithms/Algorithms/Langevin/GlobalLangevin.jl:178-179`.
- Block: `src/MCAlgorithms/Algorithms/Langevin/BlockLangevin.jl:179-180`.

This is zero-temperature descent/rejection behavior, not sampling from
`exp(-H/T)` at positive `T`.

Finally, if `_finite_derivative` replaces `NaN` or `Inf` derivatives with zero,
the proposal is no longer the Langevin proposal for the original derivative
field at that state. The Metropolis-Hastings algebra still corrects the
proposal actually used if its density is evaluated consistently, but it does
not prove that the dynamics correspond to the intended physical derivative.

Code representation:

- `_finite_derivative` is defined in
  `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:66-68`.
- It is applied to forward and reverse derivatives before those derivatives are
  used in the proposal means, e.g.
  `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:201-203` and
  `src/MCAlgorithms/Algorithms/Langevin/LocalLangevin.jl:242-243`.
