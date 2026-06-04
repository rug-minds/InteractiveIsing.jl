# Design and Performance Notes

This file records baseline MNIST manager choices that affect runtime. Keep it
focused on implementation decisions that should survive future experiment edits.

## Current Design

- Training and validation use `ProcessManager` with persistent normal `Process`
  workers.
- Each manager job is now a chunk of sample indices, not one copied sample.
- The recipe `start!` hook stores one `Threads.@spawn` task per manager slot.
  Inside that task, the slot's existing worker runs each sample in the chunk via
  `runprocessinline!`.
- The default chunk size is `ceil(minibatch_size / workers)`, so one minibatch
  normally dispatches about one chunk per worker. Override with
  `ISING_MNIST_IF_CHUNK_SIZE` when benchmarking.
- The supported scheduler for this path is `ISING_MNIST_IF_SCHEDULER=spawn`.
  The chunking already amortizes per-job task overhead; threaded schedules should
  be benchmarked separately before using them for production runs.
- The source graph and worker graphs pointer-share the sparse adjacency storage
  and the learnable base bias vector. Worker-local state, clamp vectors,
  gradient buffers, RNG state, and input-field buffers stay private.
- Field-mode image input writes the image-induced local field directly into the
  worker-local second `MagField`. It no longer copies that field into itself or
  clears/writes the input layer state on every sample.
- Validation is manager-backed too, with worker-local counters and chunked sample
  indices.

## Why These Choices Matter

- Chunking reduces manager scheduling overhead from one task per sample to about
  one task per worker per minibatch.
- Moving sample loading into the worker task keeps the main thread from copying
  input and target vectors for every example before dispatch.
- Keeping normal `Process` workers avoids relying on `InlineChunkWorker`, which
  is still changing in the local StatefulAlgorithms package.
- Sharing `J` and the base bias avoids per-worker parameter synchronization after
  each Adam update. Only the source graph receives the optimizer write; workers
  already see the shared arrays.
- Worker-local input fields avoid mutating the source graph state and avoid
  data races between workers.
- Keeping the `784 -> hidden` weights outside the sampled graph removes the
  old structural input layer from the work being relaxed each sample. The
  sampled graph now contains only hidden/output units, so both the dynamics and
  the gradient path operate on the same reduced state as the intended field
  formulation.
- Projecting the image once into a worker-local field buffer is cheaper than
  writing graph input state or rebuilding input-dependent graph state per
  sample. This also keeps the `Process` path close to the bespoke reduced-graph
  loop semantically.

## Caveats

- The sampled graph should stay reduced to hidden/output units only. The
  `784 -> hidden` weights belong to an external projection matrix that writes
  into a worker-local magnetic field buffer before each sample.
- Old `784-120-40` diagnostics from before the reduced-graph change are not
  valid for current serial Process-versus-bespoke conclusions. Those rows mixed
  in the legacy structural input layer and should not be used to argue that
  `StatefulAlgorithms` is slower.

## What Made `StatefulAlgorithms` Faster Now

- The biggest fix was architectural, not scheduler-related: the baseline no
  longer samples a structural input layer. `StatefulAlgorithms` now runs the intended
  hidden/output graph and applies the image through a worker-local magnetic
  field, which removes wasted work that the old path was still doing.
- The worker algorithm now projects the image into a reusable local buffer and
  installs that buffer into the second `MagField`, instead of touching graph
  state through the old input-layer path.
- The reduced graph keeps pointer sharing effective. Worker graphs still share
  sparse `J` and the base bias with the source graph, while only the
  image-dependent field buffer and worker state stay private.
- The benchmark scripts are now apples-to-apples with the reduced graph. The
  earlier serial claim that bespoke was about `3x` faster came from stale
  diagnostics, not from the current implementation.

## Measured Runtime Notes

- Fixed `batchsize=128` is not a clean worker-scaling diagnostic because it
  changes chunk size when worker count changes. With `workers=16`, the implicit
  chunk size is 8; with `workers=32`, it is 4.
- The cleaner diagnostic is to fix chunk size and scale `batchsize =
  workers * chunk_size`. With real timing parameters (`sweeps=500`, `β=5`,
  `temp=0.001`, `stepsize=0.5`, `relaxation_steps=80000`), the rerun fixed
  chunk-size diagnostic on `2026-05-30` gave:
  - 16 workers, chunk 4: mean `0.00321 s/example`
  - 16 workers, chunk 8: mean `0.00314 s/example`
  - 16 workers, chunk 16: mean `0.00313 s/example`
  - 32 workers, chunk 4: mean `0.00218 s/example`
  - 32 workers, chunk 8: mean `0.00232 s/example`
  - 32 workers, chunk 16: mean `0.00226 s/example`
- So the current best measured throughput point is 32 workers with chunk 4,
  and the practical conclusion is still "use 32 workers", but the gap between
  chunk sizes 4 and 16 is now modest rather than dramatic.
- The updated single-thread diagnostics now say the serial `Process` path is
  roughly on par with the bespoke reduced-graph loop, not slower by multiples:
  - Reduced bespoke direct Metropolis mean over 3 reruns:
    `0.03550 s/example`
  - Reduced normal `Process` mean over the same reruns:
    `0.01674 s/example`
  - Reduced `InlineProcess` mean over the same reruns:
    `0.01736 s/example`
- The 1-worker real training-path diagnostic (`LocalLangevin` manager code,
  measured through `runprocessinline!`) averaged `0.03530 s/example` over 8
  serial samples, while the 1-worker manager chunk path measured
  `0.04033 s/example`, about `14%` overhead over the worker-inline path.
- Current practical default for throughput runs: use `workers=32` with
  `chunk_size=4`, and treat chunk sizes 8 and 16 as nearby fallbacks if batch
  construction or load balancing makes them easier.

## Smoke Checks

- `diagnostics/runs/20260530_chunked_spawn_smoke`: one-worker validation smoke.
- `diagnostics/runs/20260530_chunked_spawn_train_smoke`: two-worker one-epoch
  smoke with very small data and relaxation settings.
- `diagnostics/runs/20260530_chunked_spawn_scaling/chunk_size_grid_16_32.jl`:
  fixed chunk-size scaling diagnostic for chunk sizes 4, 8, and 16 with 16 and
  32 workers.
