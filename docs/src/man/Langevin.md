# Langevin Algorithms

This page documents the Langevin algorithms currently implemented in
`src/MCAlgorithms/Algorithms/Langevin`.

There are three Langevin update types:

- `LocalLangevin`: proposes one single-spin move per `Processes.step!`.
- `GlobalLangevin`: proposes one all-active-spins move per internal proposal.
- `BlockLangevin`: proposes one multi-spin move on a random cyclic block of
  active spins per internal proposal.

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
- `group_steps`: repeats the algorithm's internal proposal cycle. It does not
  divide `stepsize`.
- `adjusted`: selects between Metropolis-adjusted Langevin and always-accepted
  reflected Langevin.

The returned step context includes:

- `proposal`: the last `FlipProposal` or `MultiSpinProposal`.
- `ΔE`: the energy change of the last adjusted proposal where it was computed.
- `accepted`: number of accepted proposals in this call.
- `attempted`: number of attempted proposals in this call.
- `acceptance_rate`: `accepted / attempted`.
- `T`: current model temperature.
- `η`: current effective stepsize.
- `σ`: current Gaussian noise scale.
- `group_steps`: effective group step count.
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

`GlobalLangevin` proposes a move for every active spin at once. The proposal is
stored as a `MultiSpinProposal`.

For active state vector `x_A`, the adjusted proposal is

```math
y_A = x_A - \eta\,\nabla_A H(x) + \sqrt{2\eta T}\,\xi,
\qquad \xi \sim \mathcal{N}(0, I).
```

If `adjusted=true`, the entire global proposal is rejected if any proposed
coordinate is out of bounds. Otherwise the whole vector is accepted or rejected
as one Metropolis-Hastings move.

If `adjusted=false`, each coordinate is reflected into its layer bounds, all
coordinates are written to the graph state, and the whole `MultiSpinProposal` is
marked accepted.

`group_steps` repeats this whole global proposal multiple times inside one
`Processes.step!`.

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

`BlockLangevin` is between local and global Langevin. Each proposal updates a
random cyclic block of active spins. If there are `n` active spins and
`m = min(block_size, n)`, the algorithm chooses a random cyclic offset and takes
`m` consecutive active indices modulo `n`.

The adjusted proposal and acceptance rule are the same as `GlobalLangevin`, but
restricted to the selected block.

The unadjusted proposal and reflection rule are also the same as
`GlobalLangevin`, but restricted to the selected block.

`group_steps` repeats the block proposal multiple times inside one
`Processes.step!`.

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
