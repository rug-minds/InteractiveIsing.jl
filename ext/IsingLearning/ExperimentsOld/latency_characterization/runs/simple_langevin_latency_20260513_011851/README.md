# Non-Manager Simple Langevin XOR Latency

Generated: `2026-05-13 01:20:47`

## Settings

- `workers`: `2`
- `minit`: `1`
- `eval_repeats`: `1`
- `free_relaxation`: `2`
- `nudged_relaxation`: `2`
- `temp`: `0.005`
- `stepsize`: `0.4`
- `beta`: `2.0`

## Phase Timings

| phase | seconds | compile time | recompile time | gc time | bytes |
|---|---:|---:|---:|---:|---:|
| `include simple experiment` | 9.267 | 1.4265 | 0.5178 | 0.3563 | 685587064 |
| `simple graph construction` | 1.6533 | 1.648 | 0.0264 | 0.0857 | 365840368 |
| `simple layer construction` | 0.0455 | 0.0444 | 0.0 | 0.0 | 1890928 |
| `simple init_simple_trainer` | 33.7007 | 33.6361 | 0.5428 | 0.2251 | 4072928528 |
| `simple simple_dataset` | 0.0 | 0.0 | 0.0 | 0.0 | 256 |
| `simple gradient_buffer` | 0.0 | 0.0 | 0.0 | 0.0 | 800 |
| `simple eval: evaluate_simple!` | 9.8247 | 9.8235 | 0.0 | 0.1596 | 633562528 |
| `simple train1: zero buffers` | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| `simple train1: worker loop` | 69.0218 | 87.035 | 0.0 | 0.4776 | 2422351376 |
| `simple train1: optimizer/broadcast` | 0.3045 | 0.304 | 0.0 | 0.0 | 122631296 |
| `simple train2: zero buffers` | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| `simple train2: worker loop` | 0.0005 | 0.0 | 0.0 | 0.0 | 228672 |
| `simple train2: optimizer/broadcast` | 0.0 | 0.0 | 0.0 | 0.0 | 6864 |

## Notes

- `simple train: worker loop` contains process reset, launch, wait/close, and gradient collection for all sample/repeat jobs.
- `simple eval: evaluate_simple!` contains all validation repeats and process execution.
- Large compile_time values identify first-specialization latency; a second call in the same process should isolate actual runtime.
