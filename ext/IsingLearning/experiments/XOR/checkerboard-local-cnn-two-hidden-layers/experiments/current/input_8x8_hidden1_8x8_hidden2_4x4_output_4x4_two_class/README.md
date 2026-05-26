# Input 8x8, Hidden1 8x8, Hidden2 4x4, Output 4x4

This is the current checkerboard CNN-like XOR experiment. The architecture is
fixed and only the locality radii change.

```text
8x8 checkerboard input -> 8x8 hidden1 -> 4x4 hidden2 -> 4x4 two-class output
```

The radius grid is:

- `r1 = 1, 2, 3, 4, 5` for input -> hidden1 and hidden1 recurrent locality.
- `r2 = 1, 2` for hidden1 -> hidden2, hidden2 recurrent locality, and
  hidden2 -> output readout locality.

Each `r1_*_r2_*` subfolder contains the metrics, checkpoints, and
`learning_curve.png` for one radius combination. The aggregate files in this
folder compare all combinations in the same architecture:

- `metrics.csv`: all logged epochs for all combinations.
- `summary.csv`: one best-result row per combination.
- `learning_summary.png`: learning curves plus best-margin comparison.
- `validation_1024_best_margin_params.csv` and `.png`: high-repeat validation
  of each saved best-margin checkpoint.
- `success.md`: explanation of the settings and radius choices that made the
  successful runs work.
