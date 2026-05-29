# Manager Time-Averaged XOR Latency

Generated: `2026-05-13 11:32:35`

## Settings

- `workers`: `1`
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
| `include manager experiment` | 9.5773 | 1.6144 | 0.4626 | 0.3013 | 707492888 |
| `manager init_timeavg_trainer` | 22.1088 | 22.0626 | 0.5703 | 0.2539 | 2987105536 |
| `manager simple_dataset` | 0.0 | 0.0 | 0.0 | 0.0 | 256 |
| `manager gradient_buffer` | 0.0 | 0.0 | 0.0 | 0.0 | 800 |
| `manager eval: build jobs` | 0.0 | 0.0 | 0.0 | 0.0 | 816 |
| `manager eval: sync/reinit workers` | 0.0309 | 0.0308 | 0.0 | 0.0 | 2274096 |
| `manager eval: allocate output buckets` | 0.0 | 0.0 | 0.0 | 0.0 | 224 |
| `manager eval: ProcessManager.run!` | 7.3215 | 7.3198 | 0.0 | 0.0585 | 569344416 |
| `manager eval: reduce metrics` | 0.0202 | 0.02 | 0.0 | 0.0 | 2685760 |
| `manager train1: zero buffers` | 0.1342 | 0.134 | 0.0 | 0.0 | 4281216 |
| `manager train1: build jobs` | 0.0 | 0.0 | 0.0 | 0.0 | 1152 |
| `manager train1: ProcessManager.run!` | 35.7704 | 35.7685 | 0.0 | 0.2317 | 1143124928 |
| `manager train1: optimizer update` | 0.2884 | 0.288 | 0.0 | 0.0 | 122629712 |
| `manager train1: sync/reinit workers` | 0.0103 | 0.0103 | 0.0 | 0.0 | 615088 |
| `manager warm init_timeavg_trainer` | 10.4135 | 10.387 | 0.0 | 0.0782 | 1099734464 |
| `manager warm gradient_buffer` | 0.0 | 0.0 | 0.0 | 0.0 | 800 |
| `manager eval: build jobs` | 0.0 | 0.0 | 0.0 | 0.0 | 816 |
| `manager eval: sync/reinit workers` | 0.0 | 0.0 | 0.0 | 0.0 | 32 |
| `manager eval: allocate output buckets` | 0.0 | 0.0 | 0.0 | 0.0 | 224 |
| `manager eval: ProcessManager.run!` | 0.0004 | 0.0 | 0.0 | 0.0 | 66928 |
| `manager eval: reduce metrics` | 0.0 | 0.0 | 0.0 | 0.0 | 960 |
| `manager train101: zero buffers` | 0.0826 | 0.0825 | 0.0 | 0.0 | 592400 |
| `manager train101: build jobs` | 0.0 | 0.0 | 0.0 | 0.0 | 1152 |
| `manager train101: ProcessManager.run!` | 34.5708 | 34.5694 | 0.0 | 0.3117 | 885558048 |
| `manager train101: optimizer update` | 0.0001 | 0.0 | 0.0 | 0.0 | 5616 |
| `manager train101: sync/reinit workers` | 0.01 | 0.01 | 0.0 | 0.0 | 614864 |

## Notes

- `ProcessManager.run!` contains the actual worker process launch, relaxation loop, wait/close, consume, and final flush.
- The manager training path now accumulates worker-local buffers and flushes them once at the end of the batch.
- Large compile_time values identify first-specialization latency; later epochs in the same process should have much lower compile_time.
