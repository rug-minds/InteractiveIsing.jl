# Langevin Algorithms

This page documents the Langevin algorithms currently implemented in
`src/MCAlgorithms/Algorithms/Langevin`.

There are three Langevin update types:

- `LocalLangevin`: proposes one single-spin move per `StatefulAlgorithms.step!`.
- `GlobalLangevin`: refreshes all active-spin derivatives, then proposes one
  single-spin move per `StatefulAlgorithms.step!`.
- `BlockLangevin`: refreshes derivatives on a shuffled group of active
  spins, then proposes one single-spin move per `StatefulAlgorithms.step!`.

The Boltzmann correctness argument for the adjusted algorithms is kept separate
in the developer page `Langevin Boltzmann Proof`.

## Shared Interface

All Langevin algorithms use the model temperature from `temp(model)`. There is
no separate Langevin temperature parameter.

All constructors have a `stepsize` keyword. This constructor value is the
default initial stepsize. During `StatefulAlgorithms.init`, the algorithm checks the
process context for a `:stepsize` variable. If the context provides one, that
value is used instead of the constructor default. The initialized context stores
the result as a `Ref`, so other process algorithms can share or tune it.

The stochastic noise scale is

```math
\sigma = \sqrt{2 \eta T},
```

where `η = max(stepsize[], eps(eltype(model)))` and `T = temp(model)`.

The common keywords are:

- `stepsize`: default value for the runtime stepsize `η`.
- `max_drift_fraction`: only limits the deterministic drift in the unadjusted
  path. The adjusted path uses the raw Langevin drift required by the MALA
  proposal density.
- `group_steps`: retained for compatibility with existing contexts. A
  `StatefulAlgorithms.step!` call attempts one spin proposal for each Langevin algorithm;
  it does not divide `stepsize`.
- `adjusted`: selects between Metropolis-adjusted Langevin and always-accepted
  Langevin with clamped deterministic drift and reflected stochastic
  displacement.

The returned step context includes:

- `proposal`: the last `FlipProposal`.
- `ΔE`: the energy change of the last adjusted proposal where it was computed.
- `accepted`: number of accepted proposals in this call.
- `attempted`: number of attempted proposals in this call.
- `acceptance_rate`: `accepted / attempted`.
- `T`: current model temperature.
- `η`: current effective stepsize.
- `σ`: current Gaussian noise scale.
- `group_steps`: effective group step count.
- `block_size`: selected derivative block size for block Langevin outputs.
- `refreshed_gradient`: true when this step recomputed the cached derivative
  cycle for global or block Langevin.
- `gradient_max`: maximum absolute derivative in the stored derivative buffer.
- `gradient_rms`: derivative RMS over the derivatives measured by this call,
  except `LocalLangevin`, where a step is one spin and this is the absolute
  derivative used by that one proposal.
- `reflected_fraction`: fraction of unadjusted proposals reflected back into
  bounds.

## `LocalLangevin`

```julia
LocalLangevin(;
    stepsize = 0.1,
    max_drift_fraction = 0.15,
    group_steps = 1,
    adjusted = true,
)
```

`LocalLangevin` proposes exactly one spin update per `StatefulAlgorithms.step!`.

Internally it keeps a sweep cursor:

- `sweep_position`: position inside the current active-index sweep.
- `sweep_offset`: random cyclic offset for the current sweep.
- `group_remaining`: number of remaining full sweeps in the current internal
  cycle.

When a new internal cycle starts, the algorithm computes the current derivative
for every active spin and stores it in `dH_prealloc`. It then chooses a random
cyclic offset. Each subsequent `StatefulAlgorithms.step!` advances one position in that
cyclic order and returns immediately after that single spin proposal. This keeps
the old random cyclic sweep semantics while allowing other algorithms to observe
accepted/rejected proposals after every spin update.

For the selected spin `i`, the adjusted proposal is

```math
y_i = x_i - \eta\,\partial_i H(x) + \sqrt{2\eta T}\,\xi,
\qquad \xi \sim \mathcal{N}(0,1).
```

If `adjusted=true`, out-of-bounds proposals are rejected. In-bounds proposals
use the Metropolis-Hastings/MALA acceptance probability documented in the proof
page.

If `adjusted=false`, the deterministic drift is capped to

```math
|\eta\,\partial_i H(x)| \le
\texttt{max_drift_fraction} \cdot (\texttt{high_state} - \texttt{low_state}),
```

then the deterministic drift result is clamped into bounds, the stochastic
displacement is reflected into bounds, and the proposal is accepted.

## `GlobalLangevin`

```julia
GlobalLangevin(;
    stepsize = 0.1,
    max_drift_fraction = 0.15,
    group_steps = 1,
    adjusted = false,
)
```

`GlobalLangevin` recomputes the derivative for every active spin at the start
of a proposal cycle. With `adjusted=true`, it constructs and accepts/rejects the
whole active-spin Langevin proposal at that global level. If accepted, the
accepted vector proposal is then written as one `FlipProposal` per subsequent
`StatefulAlgorithms.step!`. With `adjusted=false`, it skips the global accept/reject and
streams reflected single-spin writes from the cached derivative cycle.

For the selected spin `i`, the adjusted proposal is

```math
y_i = x_i - \eta\,\partial_i H(x) + \sqrt{2\eta T}\,\xi,
\qquad \xi \sim \mathcal{N}(0,1).
```

If `adjusted=true`, an out-of-bounds coordinate rejects the whole global
proposal. Otherwise the whole vector proposal is accepted or rejected as one
Metropolis-Hastings/MALA move.

If `adjusted=false`, the selected coordinate's deterministic drift result is
clamped into its layer bounds. The stochastic displacement is then reflected into
the bounds, written to the graph state, and marked accepted.

`group_steps` does not repeat proposals inside one `StatefulAlgorithms.step!`; each call
attempts one spin update.

## `BlockLangevin`

```julia
BlockLangevin(;
    stepsize = 0.1,
    max_drift_fraction = 0.15,
    block_size = 256,
    group_steps = 1,
    adjusted = false,
)
```

`BlockLangevin` is between local and global Langevin. At the start of a proposal
cycle it refreshes a group of active-spin derivatives from a shuffled
active-spin order. If there are `n` active spins and `m = min(block_size, n)`,
the algorithm walks that shuffled order in chunks of `m`, reshuffling when the
next chunk would overrun the order. The selected group is therefore not a
spatial block or a run of linear indices. With `adjusted=true`, it constructs
and accepts/rejects the whole block proposal at that block level, then streams
accepted block entries as one `FlipProposal` per subsequent `StatefulAlgorithms.step!`.
With `adjusted=false`, it streams always-accepted single-spin writes from the
cached block derivative cycle using clamped deterministic drift and reflected
stochastic displacement.

The adjusted proposal and acceptance rule are the same as `GlobalLangevin`, but
the derivative refresh is restricted to the selected block.

The unadjusted proposal and reflection rule are also the same as
`GlobalLangevin`.

`group_steps` does not repeat proposals inside one `StatefulAlgorithms.step!`; each call
attempts one spin update.

## Adjusted Versus Unadjusted

`adjusted=true` means the proposal is accepted with the current-gradient
Metropolis-Hastings/MALA correction at the algorithm's proposal scope:
single-spin for local, all active spins for global, and the selected block for
block Langevin. Accepted global/block vector proposals are streamed into the
graph one spin per `StatefulAlgorithms.step!` after the proposal-level accept decision.

`adjusted=false` means deterministic drift is clamped into bounds, stochastic
displacement is reflected into bounds, and the result is accepted without a
Metropolis-Hastings correction. This can be useful for interactive or relaxation
dynamics, but it is not an exact Boltzmann sampler in the current
implementation.

## Tuners

Tuners live in `src/MCAlgorithms/Algorithms/Langevin/Tuners`.

`AcceptanceRateStepSizeTuner` adapts a shared/routed `stepsize` from the last
`acceptance_rate`. It is only meaningful for `adjusted=true`. It throws if the
routed/shared `adjusted` value is false, because unadjusted Langevin accepts by
construction and therefore does not have an informative acceptance rate.

`DriftStepSizeTuner` adapts a shared/routed `stepsize` so that

```math
\texttt{stepsize} \cdot \texttt{gradient_max}
\approx
\texttt{target_drift}.
```

This is a stability/interactivity tuner. It is not a Boltzmann-correctness
criterion.
