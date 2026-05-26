# 512-Repeat Edge Checkpoint Validation

Use of this file: compact reload validation for the best-margin checkpoints from this run.

Run:

- `edge_twoclass_zero_side16_nn1to10_e160`
- architecture: `16 input edge -> 16x16 hidden -> 16 output edge`
- output mode: replicated `two_class`
- dynamics: zero-start `BlockLangevin`
- free/nudged sweeps: `20 / 20`
- workers during training: `32`
- jobs per epoch: `32`
- repeats per case during training: `64`

Reload validation used `512` evaluation repeats on the serialized `best_margin_params.bin` checkpoints:

| Config | Score MSE | Output MSE | Accuracy | Worst Margin | Scores |
|---|---:|---:|---:|---:|---|
| `nn6` | 0.049007 | 0.308099 | 1.0 | 0.690879 | `-1.11570, 1.29501, 1.00739, -0.69088` |
| `nn7` | 0.039386 | 0.271038 | 1.0 | 0.867985 | `-1.07572, 1.36599, 0.97924, -0.86798` |
| `nn9` | 0.034092 | 0.358220 | 1.0 | 0.788431 | `-0.83651, 1.09299, 1.23713, -0.78843` |

The best margin in this validation is `nn7`; the best score MSE is `nn9`.
