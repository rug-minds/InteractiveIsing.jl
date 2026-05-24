# High-Repeat Validation

Use of this file: reload validation for the successful local CNN-like XOR checkpoint.

Checkpoint:

`h8_r8_open/best_margin_params.bin`

Validation settings:

- eval repeats: `512`
- init mode: `zero`
- output mode: `two_class`
- architecture: `8x8 -> 8x8 -> 8x8 -> 4x4`
- radius: `8`

Result:

- score MSE: `0.049015723`
- physical output MSE: `0.754863`
- accuracy: `1.0`
- all correct: `true`
- worst-case margin: `0.6517369`
- mean margin: `0.8264536`
- scores: `-0.9909 | 0.92638 | 0.7368 | -0.65174`
- margins: `0.9909 | 0.92638 | 0.7368 | 0.65174`
- predictions: `0 | 1 | 1 | 0`
