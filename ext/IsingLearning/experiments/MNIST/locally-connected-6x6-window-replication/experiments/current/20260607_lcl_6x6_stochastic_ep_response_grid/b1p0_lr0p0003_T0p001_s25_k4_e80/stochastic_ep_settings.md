# LCL Stochastic-EP Diagnostic

- estimator: one-sided nudged observable average
- stochastic nudged rollouts per example: `4`
- nudging: `tangent fixed free-equilibrium error force`
- beta/temp/stepsize: `1.0` / `0.001` / `0.5`
- sweeps: `25.0`
- note: hidden-output gradients average `s_i*s_j` observables directly.
