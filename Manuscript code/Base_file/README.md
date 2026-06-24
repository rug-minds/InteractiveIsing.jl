# Physical-unit manuscript experiments

This directory now has two independent experiment entry points:

- `AnnealExperiment.jl`: temperature cycle with polarization and mobile-charge dynamics.
- `PulseExperiment.jl`: triangular field pulse followed by relaxation.

Shared model construction, physical parameters, plotting, and saving live in
`Basefile.jl`. Runtime loggers and energy diagnostics live in
`ExperimentLoggers.jl`.

The previous combined script is retained as `LegacyCombinedBasefile.jl` for
comparison.

## Run

From the repository root:

```powershell
julia --project=. "Manuscript code/Base_file/AnnealExperiment.jl"
julia --project=. "Manuscript code/Base_file/PulseExperiment.jl"
```

For a short smoke test without saving:

```powershell
$env:ISING_STEPS = "1"
$env:ISING_SAVE_OUTPUTS = "false"
julia --project=. "Manuscript code/Base_file/AnnealExperiment.jl"
```

## Physical convention

- internal energy unit: `1 meV`
- internal length unit: `1 nm`
- charge unit: elementary charge `e`
- dipole unit: `e nm`
- state/polarization variable: dimensionless
- Monte Carlo temperature input: `k_B T` in energy units

The default `k_B T = 0.15 meV` corresponds to approximately `1.74 K`.
Landau local-potential arrays remain dimensionless multipliers of the explicit
`1 meV` polynomial energy prefactor.
