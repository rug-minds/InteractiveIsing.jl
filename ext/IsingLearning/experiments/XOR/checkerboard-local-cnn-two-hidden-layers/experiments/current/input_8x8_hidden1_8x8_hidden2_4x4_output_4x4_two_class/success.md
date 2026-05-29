# What Made This Run Work

The successful checkerboard setup is the fixed architecture:

```text
8x8 checkerboard input -> 8x8 hidden1 -> 4x4 hidden2 -> 4x4 two-class output
```

The important change from the old checkerboard folder is that the two locality
radii are separated. The best runs all used `r2 = 2`, while `r2 = 1` either
learned weakly or did not produce a useful score MSE.

## Best Configurations

The two useful 100-epoch training results were:

- `r1_4_r2_2`: best training MSE `0.04382591`, first all-correct epoch `50`,
  best minimum margin `0.72811705`.
- `r1_1_r2_2`: best training MSE `0.048765138`, first all-correct epoch `30`,
  best minimum margin `0.90207064`.

The 1024-repeat validation of `best_margin_params.bin` confirmed both:

- `r1_1_r2_2`: validation MSE `0.02229712`, accuracy `1.0`, minimum margin
  `0.78118193`.
- `r1_4_r2_2`: validation MSE `0.04172624`, accuracy `1.0`, minimum margin
  `0.71936023`.

`r1_1_r2_2` is the cleaner checkpoint to reuse first because it had the lowest
validated MSE and the strongest validated worst-case margin.

## Working Recipe

- Use `two_class` output, not scalar majority output.
- Use zero initialization before each free/nudged trajectory.
- Use `BlockLangevin` dynamics with 20 free sweeps and 20 nudged sweeps.
- Use 32 manager workers with 64 repeats per XOR case.
- Use Adam with `lr = 0.002`, `lr_decay = 0.995`, `lr_min = 0.0002`, and
  coupling weight decay `1e-4`.
- Keep hidden2/readout locality at `r2 = 2` for this architecture.

## What Did Not Work As Well

The `r2 = 1` side of the grid was too narrow for robust compression into the
`4x4` hidden2 layer. Some `r2 = 1` configurations reached full accuracy at
logged epochs, but their MSEs stayed high and their margins were small or
unstable under 1024-repeat validation.

Increasing `r1` alone did not solve the problem. The useful separation is not
"larger radius everywhere"; it is enough locality in the compressed hidden2 and
readout stage.
