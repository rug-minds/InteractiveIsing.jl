# Non-Manager Simple Langevin XOR Latency

Generated: `2026-05-13 01:56:25`

## Settings

- `workers`: `8`
- `minit`: `8`
- `eval_repeats`: `16`
- `free_relaxation`: `600`
- `nudged_relaxation`: `600`
- `temp`: `0.005`
- `stepsize`: `0.4`
- `beta`: `2.0`

## Phase Timings

| phase | seconds | compile time | recompile time | gc time | bytes |
|---|---:|---:|---:|---:|---:|
| `include simple experiment` | 9.1366 | 1.4143 | 0.433 | 0.3447 | 685650744 |
| `simple graph construction` | 1.6677 | 1.6625 | 0.0264 | 0.0956 | 365837584 |
| `simple layer construction` | 0.0453 | 0.0442 | 0.0 | 0.0 | 1891184 |
| `simple init_simple_trainer` | 107.7394 | 107.5161 | 0.5015 | 0.9061 | 12005208688 |
| `simple simple_dataset` | 0.0 | 0.0 | 0.0 | 0.0 | 256 |
| `simple gradient_buffer` | 0.0 | 0.0 | 0.0 | 0.0 | 800 |
| `simple eval: evaluate_simple!` | 9.3157 | 9.3066 | 0.0 | 0.1808 | 633947216 |
| `simple train1: zero buffers` | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| `simple train1: worker loop` | 256.1764 | 322.2362 | 0.0 | 1.8024 | 8737778256 |
| `simple train1: optimizer/broadcast` | 0.2629 | 0.2625 | 0.0 | 0.0 | 122769184 |
| `simple warm graph construction` | 0.026 | 0.0249 | 0.0 | 0.0 | 748352 |
| `simple warm layer construction` | 0.024 | 0.0232 | 0.0 | 0.0 | 750208 |
| `simple warm init_simple_trainer` | 105.5663 | 105.3048 | 0.0 | 0.8306 | 10863347696 |
| `simple warm gradient_buffer` | 0.0 | 0.0 | 0.0 | 0.0 | 800 |
| `simple eval: evaluate_simple!` | 7.6634 | 7.6545 | 0.0 | 0.0728 | 384816032 |
| `simple train101: zero buffers` | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| `simple train101: worker loop` | 257.9488 | 312.8143 | 0.0 | 1.6269 | 8421158288 |
| `simple train101: optimizer/broadcast` | 0.0094 | 0.0093 | 0.0 | 0.0 | 182160 |

## Notes

- `simple train: worker loop` contains process reset, launch, wait/close, and gradient collection for all sample/repeat jobs.
- `simple eval: evaluate_simple!` contains all validation repeats and process execution.
- Large compile_time values identify first-specialization latency; a second call in the same process should isolate actual runtime.
