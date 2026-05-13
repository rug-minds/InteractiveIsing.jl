# Non-Manager Simple Langevin XOR Latency

Generated: `2026-05-13 01:26:59`

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
| `include simple experiment` | 9.4251 | 1.4895 | 0.443 | 0.3216 | 685582984 |
| `simple graph construction` | 1.8112 | 1.8055 | 0.0293 | 0.1385 | 365838352 |
| `simple layer construction` | 0.0465 | 0.0453 | 0.0 | 0.0 | 1889840 |
| `simple init_simple_trainer` | 34.1336 | 34.0688 | 0.5558 | 0.2286 | 4072819200 |
| `simple simple_dataset` | 0.0 | 0.0 | 0.0 | 0.0 | 256 |
| `simple gradient_buffer` | 0.0 | 0.0 | 0.0 | 0.0 | 800 |
| `simple eval: evaluate_simple!` | 10.2232 | 10.2219 | 0.0 | 0.151 | 633575760 |
| `simple train1: zero buffers` | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| `simple train1: worker loop` | 68.6459 | 86.2689 | 0.0 | 0.5094 | 2422361344 |
| `simple train1: optimizer/broadcast` | 0.3011 | 0.3007 | 0.0 | 0.0 | 122713280 |
| `simple warm graph construction` | 0.0275 | 0.0265 | 0.0 | 0.0 | 748288 |
| `simple warm layer construction` | 0.0241 | 0.0234 | 0.0 | 0.0 | 749952 |
| `simple warm init_simple_trainer` | 29.1076 | 29.0413 | 0.0 | 0.2187 | 2936269600 |
| `simple warm gradient_buffer` | 0.0 | 0.0 | 0.0 | 0.0 | 800 |
| `simple eval: evaluate_simple!` | 7.8072 | 7.8062 | 0.0 | 0.1428 | 384593760 |
| `simple train101: zero buffers` | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| `simple train101: worker loop` | 66.7188 | 75.4113 | 0.0 | 0.4192 | 2108853984 |
| `simple train101: optimizer/broadcast` | 0.0076 | 0.0075 | 0.0 | 0.0 | 179664 |

## Notes

- `simple train: worker loop` contains process reset, launch, wait/close, and gradient collection for all sample/repeat jobs.
- `simple eval: evaluate_simple!` contains all validation repeats and process execution.
- Large compile_time values identify first-specialization latency; a second call in the same process should isolate actual runtime.
