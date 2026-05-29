# Langevin Checkerboard XOR Experiments

This folder contains Langevin-only checkerboard XOR experiment drivers. The
input is physical: bit A freezes one checkerboard mask to `+1`, bit B freezes
the complementary mask to `+1`. There is no four-case one-hot input code.

## File

- `langevin_checkerboard_xor.jl` runs continuous `BlockLangevin` experiments
  using the shared local checkerboard training machinery.
- Outputs are written under `runs/langevin_checkerboard_xor_<timestamp>/`.
- Each run writes a metrics CSV, a progress PNG, a summary CSV, parameter SVGs,
  and best-graph JLD2 files.

## Current Best Result

The best run so far is:

```text
runs/langevin_checkerboard_xor_20260510_035031/
```

Configuration:

```text
lgv_4x4_global4
4x4 input -> 4x4 hidden -> 4x4 output
continuous spins, zero initialization
BlockLangevin(adjusted=false, stepsize=0.05, block_size=16)
T = 0.005
free/nudged relaxation = 1000/1000
Minit = 4
β = 1.0
lr = 0.01
inter-layer scale = 0.25
in-layer scales = 0.08
```

Result:

```text
best MSE = 0.333032
accuracy = 1.0
scores = [-0.7076, 0.4788, 0.6668, -0.0705]
adjacency symmetry error after loading saved best graph = 0.0
```

This is a genuine physical-checkerboard Langevin result, but not yet a solved
low-MSE result. The `(1,1)` false case has the correct sign but a weak readout.

## Larger-Graph Attempts

Using the same stronger Langevin settings:

```text
runs/langevin_checkerboard_xor_20260510_033133/
```

- `lgv_8x8_global8`: best MSE `0.997757`, accuracy `0.5`.
- `lgv_8x8_inlaid4`: best MSE `0.994584`, accuracy `0.75`.

Scaling `β` to `16` for the inlaid 4x4 readout inside an 8x8 layer did not fix
the signal:

```text
runs/langevin_checkerboard_xor_20260510_034002/
lgv_8x8_inlaid4: best MSE = 0.996241, accuracy = 0.75
```

A larger hidden layer with the same 4x4 physical code also did not help yet:

```text
runs/langevin_checkerboard_xor_20260510_034547/
lgv_4x4_hidden8: best MSE = 0.955206, accuracy = 0.75
```

Increasing 8x8 in-layer connectivity from `internal_nn = 2` to `internal_nn = 5`
and increasing the inter-layer radius to `3.05` also did not solve the larger
cases:

```text
runs/langevin_checkerboard_xor_20260510_110107/
lgv_8x8_global8: best MSE = 0.987141, accuracy = 0.5
lgv_8x8_inlaid4: best MSE = 0.995029, accuracy = 0.5
```

The gradients were nonzero, but the learned scalar output scores remained small
instead of coherently polarizing the checkerboard output code.

After that, a broader inlaid-8x8 Langevin sweep was added in
`langevin_checkerboard_sweep.jl`.

First sweep:

```text
runs/langevin_checkerboard_sweep_20260510_114742/
internal_nn = 5
inter_radius = 3.05
inter-layer scale = 0.25
in-layer scales = 0.08
T in {0.001, 0.005, 0.02}
stepsize in {0.03, 0.08}
relaxation in {250, 1000}
```

Best point:

```text
T = 0.02, stepsize = 0.08, relaxation = 1000
best MSE = 0.985498, accuracy = 0.5
```

Second sweep with stronger couplings:

```text
runs/langevin_checkerboard_sweep_20260510_123816/
internal_nn = 5
inter_radius = 3.05
inter-layer scale = 0.5
in-layer scales = 0.15
lr = 0.005
T in {0.005, 0.02}
stepsize in {0.05, 0.1}
relaxation in {500, 2000}
```

Best point:

```text
T = 0.02, stepsize = 0.1, relaxation = 2000
best MSE = 0.879636, accuracy = 0.75
```

Extending that best point with `Minit = 4`, `eval_repeats = 8`, and `800`
epochs did not improve it:

```text
runs/langevin_checkerboard_sweep_20260510_131058/
best MSE = 0.981495, accuracy = 0.75
final MSE = 1.040189, accuracy = 0.5
```

This does not prove 8x8 cannot work. It only says that the tested larger
connectivity/temperature/stepsize/relaxation regimes did not yet produce a
stable XOR readout. The stronger-coupling `T=0.02, stepsize=0.1,
relaxation=2000` region is the first regime where the inlaid 8x8 readout moved
meaningfully at all.

## Interpretation

The 4x4 physical checkerboard case can learn with Langevin when the interaction
scale is high enough. The previous low-scale run at `T=0.001` only reached MSE
around `0.93`; increasing to `T=0.005` and stronger in/inter-layer couplings
made the readout move.

The larger checkerboard runs currently fail because the scalar distributed
readout remains near zero. Increasing `β` alone was not enough. The likely
bottleneck is not only sampler temperature but coherent polarization of a
larger output code under a normalized scalar clamp.

Next things worth trying:

- Make the output clamp act on a vector code instead of one scalar average.
- Train with a smaller output code embedded in a larger output layer, but use a
  non-normalized or explicitly scaled readout Hamiltonian in the experiment.
- Add a learning-rate schedule or best-restore loop for the 4x4 case, because
  long runs can overshoot after the best epoch.
- Try a separate temperature scale per graph based on `max_local_interaction_energy`.
