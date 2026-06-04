# Backend Update Contrastive Learning Retest

This folder reruns the same warmed 5-repeat **full contrastive learning**
benchmark as `20260530_145920_backend_update_local_langevin_retest`, but
against the newer backend package state visible in the workspace history:

- `c55be32e Pull deps/StatefulAlgorithms subtree`
- `60feb1c9 Squashed 'deps/StatefulAlgorithms/' changes from da5f10e81..fb7bb3f2b`

## Mean Result Over 5 Repeats

- direct bespoke: `0.01809 s/example`
- serial `Process`: `0.02821 s/example`
- 1-worker manager: `0.02890 s/example`

## Comparison To The Previous 5-Repeat Run

- direct bespoke: essentially unchanged (`~0.2%` faster)
- serial `Process`: slower by about `18.4%`
- 1-worker manager: slower by about `17.5%`

## Gap To Bespoke

- previous serial `Process` gap: about `1.31x`
- new serial `Process` gap: about `1.56x`
- previous 1-worker manager gap: about `1.36x`
- new 1-worker manager gap: about `1.60x`

So this backend update regressed the full contrastive learning benchmark rather
than improving it.
