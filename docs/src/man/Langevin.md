# Langevin Algorithms

This page documents the Langevin algorithms currently implemented in
`src/MCAlgorithms/Algorithms/Langevin`.

There are three Langevin update types:

- `LocalLangevin`: proposes one single-spin move per `Processes.step!`.
- `GlobalLangevin`: refreshes all active-spin derivatives, then proposes one
  single-spin move per `Processes.step!`.
- `BlockLangevin`: refreshes derivatives on a random cyclic block of active
  spins, then proposes one single-spin move per `Processes.step!`.

The Boltzmann correctness argument for the adjusted algorithms is kept separate
in the developer page `Langevin Boltzmann Proof`.

## Shared Interface

All Langevin algorithms use the model temperature from `temp(model)`. There is
no separate Langevin temperature parameter.

All constructors have a `stepsize` keyword. This constructor value is the
default initial stepsize. During `Processes.init`, the algorithm checks the
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
  `Processes.step!` call attempts one spin proposal for each Langevin algorithm;
  it does not divide `stepsize`.
- `adjusted`: selects between Metropolis-adjusted Langevin and always-accepted
  reflected Langevin.

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

`LocalLangevin` proposes exactly one spin update per `Processes.step!`.

Internally it keeps a sweep cursor:

- `sweep_position`: position inside the current active-index sweep.
- `sweep_offset`: random cyclic offset for the current sweep.
- `group_remaining`: number of remaining full sweeps in the current internal
  cycle.

When a new internal cycle starts, the algorithm computes the current derivative
for every active spin and stores it in `dH_prealloc`. It then chooses a random
cyclic offset. Each subsequent `Processes.step!` advances one position in that
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

then the proposal is reflected into bounds and accepted.

## `GlobalLangevin`

```julia
GlobalLangevin(;
    stepsize = 0.1,
    max_drift_fraction = 0.15,
    group_steps = 1,
    adjusted = false,
)
```

`GlobalLangevin` recomputes the derivative for every active spin, then selects
one active spin and proposes a single-spin move. The proposal is stored as a
`FlipProposal`.

For the selected spin `i`, the adjusted proposal is

```math
y_i = x_i - \eta\,\partial_i H(x) + \sqrt{2\eta T}\,\xi,
\qquad \xi \sim \mathcal{N}(0,1).
```

If `adjusted=true`, an out-of-bounds proposal is rejected. Otherwise the
single-spin proposal is accepted or rejected as one Metropolis-Hastings move.

If `adjusted=false`, the selected coordinate is reflected into its layer bounds,
written to the graph state, and marked accepted.

`group_steps` does not repeat proposals inside one `Processes.step!`; each call
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

`BlockLangevin` is between local and global Langevin. Each step refreshes a
random cyclic block of active-spin derivatives. If there are `n` active spins
and `m = min(block_size, n)`, the algorithm chooses a random cyclic offset and
takes `m` consecutive active indices modulo `n`. It then proposes one spin from
that block.

The adjusted proposal and acceptance rule are the same as `GlobalLangevin`, but
the derivative refresh is restricted to the selected block.

The unadjusted proposal and reflection rule are also the same as
`GlobalLangevin`.

`group_steps` does not repeat proposals inside one `Processes.step!`; each call
attempts one spin update.

## Adjusted Versus Unadjusted

`adjusted=true` means the proposal is a Metropolis-adjusted Langevin proposal.
For positive temperature, finite derivatives, exact `ΔH`, and fixed stepsize,
this is the Boltzmann-correct path.

`adjusted=false` means the proposal is reflected into bounds and accepted
without a Metropolis-Hastings correction. This can be useful for interactive or
relaxation dynamics, but it is not an exact Boltzmann sampler in the current
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
