# Backend Update Contrastive Learning Retest

This folder reruns the same reduced-baseline **full contrastive learning**
benchmark as `20260530_135809_reduced_baseline_benchmark_refresh`, but against
the newer backend package state in the current workspace.

The benchmark is not a dynamics-only microbenchmark. It measures one warmed
minibatch of the current learning example:

- free phase
- positive nudged phase
- gradient accumulation
- minibatch normalization
- one Adam update

The current learning example uses `LocalLangevin` dynamics internally, which is
why the script name still contains `local_langevin`.

## Mean Result Over 3 Reruns

- direct bespoke: `0.01785 s/example`
- serial `Process`: `0.02326 s/example`
- 1-worker manager: `0.02348 s/example`

## Comparison To The Earlier 3-Run Refresh

- direct bespoke improved by about `3.1%`
- serial `Process` improved by about `12.8%`
- 1-worker manager improved by about `27.7%`

## Gap To Bespoke

- old serial `Process` gap: about `1.43x`
- new serial `Process` gap: about `1.30x`
- old 1-worker manager gap: about `1.63x`
- new 1-worker manager gap: about `1.32x`
