# Nudgeless covariance learning rule

This experiment keeps the `784 -> 120 hidden -> 40 output` MNIST baseline, but
replaces the nudged phase with a covariance estimate from samples of the
un-nudged (`beta = 0`) model.

For a fixed input image `x`, let the free model be

```math
p_0(s \mid x; \theta) = Z_0^{-1} \exp(-E_\theta(s, x) / T).
```

Use the quadratic output cost

```math
C(s, y) = \frac{1}{2}\sum_{k \in output} (s_k - y_k)^2.
```

If the nudged distribution is written with the user's sign convention

```math
E_\beta(s) = E_\theta(s, x) - \beta C(s, y),
```

then

```math
\frac{\partial}{\partial \beta} \langle A(s) \rangle_\beta \bigg|_{\beta=0}
    = \frac{1}{T}\operatorname{cov}_0(A(s), C(s, y)).
```

The repo's direct clamping term is implemented as `E + beta*C`, so the same
state-response derivative would carry the opposite sign there. The nudgeless
estimator below does not run that phase; it samples `p0` and uses the covariance
identity directly.

The supervised objective is the free expected cost

```math
L(\theta) = \langle C(s, y) \rangle_0.
```

For parameters that affect only the model energy,

```math
\frac{\partial L}{\partial \theta}
    = -\frac{1}{T}\operatorname{cov}_0\left(C(s, y),
        \frac{\partial E_\theta(s, x)}{\partial \theta}\right).
```

For the bilinear coupling term used here,

```math
E_J(s) = -\frac{1}{2}\sum_{ij}J_{ij}s_i s_j,
```

so

```math
\frac{\partial L}{\partial J_{ij}}
    = \frac{1}{2T}\operatorname{cov}_0(C, s_i s_j).
```

For the base magnetic field and external image-to-hidden weights,

```math
E_b(s) = -\sum_i b_i s_i,
\quad
E_W(s, x) = -\sum_h \left(\sum_a W_{ha}x_a\right)s_h,
```

which gives

```math
\frac{\partial L}{\partial b_i}
    = \frac{1}{T}\operatorname{cov}_0(C, s_i),
\quad
\frac{\partial L}{\partial W_{ha}}
    = \frac{x_a}{T}\operatorname{cov}_0(C, s_h).
```

The implementation estimates these covariances by storing `Ns` beta-zero samples
after a burn-in for each `(x, y)` training example. Each stored sample records:

- `C(s, y)`
- `dE/dJ = -s_i s_j / 2`
- `dE/db = -s_i`
- `dE/dW = -x_a s_h`

The per-example gradient is then computed from the stored sample arrays as
`-cov(C, dE/dtheta)`. The manager divides the minibatch sum by
`temperature * nsamples`.

Between stored samples the worker can perturb the state with Gaussian noise and
then run a short high-stepsize Langevin segment. This is only a mixing aid. The
stored sample is taken after relaxation, not immediately after the kick, because
the covariance identity assumes beta-zero equilibrium samples rather than samples
from the artificial perturbation distribution.
