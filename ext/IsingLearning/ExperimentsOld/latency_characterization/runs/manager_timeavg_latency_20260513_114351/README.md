# Manager Time-Averaged XOR Latency

Generated: `2026-05-13 11:44:51`

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
| `include manager experiment` | 9.7229 | 1.5797 | 0.4274 | 0.3686 | 707453448 |
| `manager init_timeavg_trainer` | 34.5282 | 71.1819 | 0.562 | 0.4368 | 4249824960 |
| `manager simple_dataset` | 0.0 | 0.0 | 0.0 | 0.0 | 256 |
| `manager gradient_buffer` | 0.0 | 0.0 | 0.0 | 0.0 | 800 |
| `manager eval: build jobs` | 0.0 | 0.0 | 0.0 | 0.0 | 816 |
| `manager eval: sync/reinit workers` | 0.0295 | 0.0294 | 0.0 | 0.0 | 2278528 |
| `manager eval: allocate output buckets` | 0.0 | 0.0 | 0.0 | 0.0 | 224 |
| `manager eval: ProcessManager.run!` | 1.7838 | 1.7823 | 0.0 | 0.0114 | 119859872 |
| `manager eval: reduce metrics` | 0.0199 | 0.0198 | 0.0 | 0.0 | 2685760 |
| `manager train1: zero buffers` | 0.1348 | 0.1346 | 0.0 | 0.0 | 3341600 |
| `manager train1: build jobs` | 0.0 | 0.0 | 0.0 | 0.0 | 1152 |
| `manager train1: ProcessManager.run!` | 1.9223 | 1.9203 | 0.0 | 0.0176 | 155372896 |
| `manager train1: optimizer update` | 0.3017 | 0.3012 | 0.0 | 0.0109 | 122684944 |
| `manager train1: sync/reinit workers` | 0.01 | 0.0099 | 0.0 | 0.0 | 615456 |
| `manager warm init_timeavg_trainer` | 19.416 | 39.12 | 0.0 | 0.129 | 1663337728 |
| `manager warm gradient_buffer` | 0.0 | 0.0 | 0.0 | 0.0 | 800 |
| `manager eval: build jobs` | 0.0 | 0.0 | 0.0 | 0.0 | 816 |
| `manager eval: sync/reinit workers` | 0.0 | 0.0 | 0.0 | 0.0 | 32 |
| `manager eval: allocate output buckets` | 0.0 | 0.0 | 0.0 | 0.0 | 224 |
| `manager eval: ProcessManager.run!` | 0.0004 | 0.0 | 0.0 | 0.0 | 66608 |
| `manager eval: reduce metrics` | 0.0 | 0.0 | 0.0 | 0.0 | 960 |
| `manager train101: zero buffers` | 0.0836 | 0.0836 | 0.0 | 0.0 | 564608 |
| `manager train101: build jobs` | 0.0 | 0.0 | 0.0 | 0.0 | 1152 |
| `manager train101: ProcessManager.run!` | 1.6553 | 1.6536 | 0.0 | 0.0103 | 82885056 |
| `manager train101: optimizer update` | 0.0001 | 0.0 | 0.0 | 0.0 | 5616 |
| `manager train101: sync/reinit workers` | 0.0115 | 0.0115 | 0.0 | 0.0 | 615232 |

## Notes

- `ProcessManager.run!` contains the actual worker process launch, relaxation loop, wait/close, consume, and final flush.
- The manager training path now accumulates worker-local buffers and flushes them once at the end of the batch.
- Large compile_time values identify first-specialization latency; later epochs in the same process should have much lower compile_time.
