# Manager Time-Averaged XOR Latency

Generated: `2026-05-13 01:43:21`

## Settings

- `workers`: `8`
- `minit`: `8`
- `eval_repeats`: `16`
- `free_relaxation`: `600`
- `nudged_relaxation`: `600`
- `eval_burnin_sweeps`: `600`
- `eval_average_sweeps`: `50`
- `temp`: `0.005`
- `stepsize`: `0.4`
- `beta`: `2.0`

## Phase Timings

| phase | seconds | compile time | recompile time | gc time | bytes |
|---|---:|---:|---:|---:|---:|
| `include manager experiment` | 9.2807 | 1.4969 | 0.3964 | 0.343 | 707253672 |
| `manager init_timeavg_trainer` | 119.1902 | 118.9349 | 0.5175 | 1.0936 | 13897402832 |
| `manager simple_dataset` | 0.0 | 0.0 | 0.0 | 0.0 | 256 |
| `manager gradient_buffer` | 0.0 | 0.0 | 0.0 | 0.0 | 800 |
| `manager eval: build jobs` | 0.0 | 0.0 | 0.0 | 0.0 | 11936 |
| `manager eval: sync/reinit workers` | 0.1245 | 0.1238 | 0.0 | 0.0 | 13868384 |
| `manager eval: allocate output buckets` | 0.0 | 0.0 | 0.0 | 0.0 | 224 |
| `manager eval: ProcessManager.run!` | 40.5779 | 40.6864 | 0.0 | 0.4482 | 2614910480 |
| `manager eval: reduce metrics` | 0.0191 | 0.0189 | 0.0 | 0.0 | 2686272 |
| `manager train1: zero buffers` | 1.1637 | 1.1632 | 0.0 | 0.0 | 7195696 |
| `manager train1: build jobs` | 0.0 | 0.0 | 0.0 | 0.0 | 6672 |
| `manager train1: ProcessManager.run!` | 182.5704 | 228.413 | 0.0 | 1.4302 | 6557311664 |
| `manager train1: optimizer update` | 0.3043 | 0.3039 | 0.0 | 0.0112 | 122717504 |
| `manager train1: sync/reinit workers` | 0.1192 | 0.1186 | 0.0 | 0.0 | 8286080 |
| `manager warm init_timeavg_trainer` | 115.674 | 115.3739 | 0.0 | 0.9 | 12271647904 |
| `manager warm gradient_buffer` | 0.0 | 0.0 | 0.0 | 0.0 | 800 |
| `manager eval: build jobs` | 0.0 | 0.0 | 0.0 | 0.0 | 11936 |
| `manager eval: sync/reinit workers` | 0.1427 | 0.1418 | 0.0 | 0.0 | 13823584 |
| `manager eval: allocate output buckets` | 0.0 | 0.0 | 0.0 | 0.0 | 224 |
| `manager eval: ProcessManager.run!` | 37.7604 | 37.8467 | 0.0 | 0.4341 | 2330122560 |
| `manager eval: reduce metrics` | 0.0 | 0.0 | 0.0 | 0.0 | 1472 |
| `manager train101: zero buffers` | 1.1528 | 1.1524 | 0.0 | 0.0 | 3508192 |
| `manager train101: build jobs` | 0.0 | 0.0 | 0.0 | 0.0 | 6672 |
| `manager train101: ProcessManager.run!` | 178.0223 | 231.5608 | 0.0 | 1.0148 | 5977812832 |
| `manager train101: optimizer update` | 0.0001 | 0.0 | 0.0 | 0.0 | 5616 |
| `manager train101: sync/reinit workers` | 0.1214 | 0.1207 | 0.0 | 0.0 | 8280160 |

## Notes

- `ProcessManager.run!` contains the actual worker process launch, relaxation loop, wait/close, consume, and final flush.
- The manager training path now accumulates worker-local buffers and flushes them once at the end of the batch.
- Large compile_time values identify first-specialization latency; later epochs in the same process should have much lower compile_time.
