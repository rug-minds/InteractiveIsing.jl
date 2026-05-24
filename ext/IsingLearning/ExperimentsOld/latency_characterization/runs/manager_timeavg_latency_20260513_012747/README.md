# Manager Time-Averaged XOR Latency

Generated: `2026-05-13 01:31:09`

## Settings

- `workers`: `2`
- `minit`: `1`
- `eval_repeats`: `1`
- `free_relaxation`: `2`
- `nudged_relaxation`: `2`
- `eval_burnin_sweeps`: `1`
- `eval_average_sweeps`: `2`
- `temp`: `0.005`
- `stepsize`: `0.4`
- `beta`: `2.0`

## Phase Timings

| phase | seconds | compile time | recompile time | gc time | bytes |
|---|---:|---:|---:|---:|---:|
| `include manager experiment` | 10.2688 | 1.5275 | 0.4342 | 0.4264 | 707244600 |
| `manager init_timeavg_trainer` | 37.9078 | 37.8345 | 0.5463 | 0.4279 | 4706631104 |
| `manager simple_dataset` | 0.0 | 0.0 | 0.0 | 0.0 | 256 |
| `manager gradient_buffer` | 0.0 | 0.0 | 0.0 | 0.0 | 800 |
| `manager eval: build jobs` | 0.0 | 0.0 | 0.0 | 0.0 | 816 |
| `manager eval: sync/reinit workers` | 0.051 | 0.0509 | 0.0 | 0.0 | 4498416 |
| `manager eval: allocate output buckets` | 0.0 | 0.0 | 0.0 | 0.0 | 224 |
| `manager eval: ProcessManager.run!` | 11.669 | 11.7212 | 0.0 | 0.0917 | 859447248 |
| `manager eval: reduce metrics` | 0.019 | 0.0188 | 0.0 | 0.0 | 2685760 |
| `manager train1: zero buffers` | 0.3312 | 0.331 | 0.0 | 0.0 | 4673760 |
| `manager train1: build jobs` | 0.0 | 0.0 | 0.0 | 0.0 | 1152 |
| `manager train1: ProcessManager.run!` | 66.7977 | 84.0551 | 0.0 | 0.5112 | 2535332176 |
| `manager train1: optimizer update` | 0.318 | 0.3176 | 0.0 | 0.0 | 122645216 |
| `manager train1: sync/reinit workers` | 0.0264 | 0.0264 | 0.0 | 0.0 | 1504880 |
| `manager warm init_timeavg_trainer` | 27.8957 | 27.8355 | 0.0 | 0.2234 | 3086065312 |
| `manager warm gradient_buffer` | 0.0 | 0.0 | 0.0 | 0.0 | 800 |
| `manager eval: build jobs` | 0.0 | 0.0 | 0.0 | 0.0 | 816 |
| `manager eval: sync/reinit workers` | 0.0526 | 0.0524 | 0.0 | 0.0 | 4497904 |
| `manager eval: allocate output buckets` | 0.0 | 0.0 | 0.0 | 0.0 | 224 |
| `manager eval: ProcessManager.run!` | 9.9061 | 9.9722 | 0.0 | 0.1563 | 585602896 |
| `manager eval: reduce metrics` | 0.0 | 0.0 | 0.0 | 0.0 | 960 |
| `manager train101: zero buffers` | 0.2975 | 0.2974 | 0.0 | 0.0 | 985472 |
| `manager train101: build jobs` | 0.0 | 0.0 | 0.0 | 0.0 | 1152 |
| `manager train101: ProcessManager.run!` | 46.3125 | 76.4598 | 0.0 | 0.3341 | 1744279168 |
| `manager train101: optimizer update` | 0.0001 | 0.0 | 0.0 | 0.0 | 5616 |
| `manager train101: sync/reinit workers` | 0.0269 | 0.0268 | 0.0 | 0.0 | 1848912 |

## Notes

- `ProcessManager.run!` contains the actual worker process launch, relaxation loop, wait/close, consume, and final flush.
- The manager training path now accumulates worker-local buffers and flushes them once at the end of the batch.
- Large compile_time values identify first-specialization latency; later epochs in the same process should have much lower compile_time.
