# Manager Time-Averaged XOR Latency

Generated: `2026-05-13 01:18:07`

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
| `include manager experiment` | 11.2567 | 1.8388 | 0.4568 | 0.5318 | 708113880 |
| `manager init_timeavg_trainer` | 37.8649 | 37.788 | 0.5887 | 0.3645 | 4706492496 |
| `manager simple_dataset` | 0.0 | 0.0 | 0.0 | 0.0 | 256 |
| `manager gradient_buffer` | 0.0 | 0.0 | 0.0 | 0.0 | 800 |
| `manager eval: build jobs` | 0.0 | 0.0 | 0.0 | 0.0 | 816 |
| `manager eval: sync/reinit workers` | 0.0592 | 0.059 | 0.0 | 0.0 | 4498704 |
| `manager eval: allocate output buckets` | 0.0 | 0.0 | 0.0 | 0.0 | 224 |
| `manager eval: ProcessManager.run!` | 12.6637 | 12.7183 | 0.0 | 0.1781 | 859365616 |
| `manager eval: reduce metrics` | 0.0182 | 0.0181 | 0.0 | 0.0 | 2685760 |
| `manager train1: zero buffers` | 0.342 | 0.3418 | 0.0 | 0.0 | 4673760 |
| `manager train1: build jobs` | 0.0 | 0.0 | 0.0 | 0.0 | 1152 |
| `manager train1: ProcessManager.run!` | 70.4592 | 88.4276 | 0.0 | 0.5536 | 2539719584 |
| `manager train1: optimizer update` | 0.2923 | 0.2918 | 0.0 | 0.0 | 122645216 |
| `manager train1: sync/reinit workers` | 0.0285 | 0.0284 | 0.0 | 0.0 | 1504880 |
| `manager train2: zero buffers` | 0.0 | 0.0 | 0.0 | 0.0 | 8096 |
| `manager train2: build jobs` | 0.0 | 0.0 | 0.0 | 0.0 | 1152 |
| `manager train2: ProcessManager.run!` | 0.0004 | 0.0 | 0.0 | 0.0 | 165472 |
| `manager train2: optimizer update` | 0.0 | 0.0 | 0.0 | 0.0 | 5616 |
| `manager train2: sync/reinit workers` | 0.0 | 0.0 | 0.0 | 0.0 | 64 |

## Notes

- `ProcessManager.run!` contains the actual worker process launch, relaxation loop, wait/close, consume, and final flush.
- The manager training path now accumulates worker-local buffers and flushes them once at the end of the batch.
- Large compile_time values identify first-specialization latency; later epochs in the same process should have much lower compile_time.
