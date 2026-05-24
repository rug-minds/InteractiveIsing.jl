# Edge Signal XOR Grid

This grid follows the first `NN=1` and `NN=5` runs.

What is being tested:
- stronger input/output edge couplings: `0.12` to `0.25` instead of `0.08`
- smaller hidden-local coupling when `NN` is larger
- temperature fractions `0.015`, `0.025`, `0.05`, and `0.08` of max column interaction
- non-periodic hidden boundaries

The main failure mode to watch is output means collapsing back toward zero or all cases sharing the same sign.

Number of configs: `12`
