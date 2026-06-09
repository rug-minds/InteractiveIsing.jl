# Ising vs XY Response Notes

This note records the current working hypothesis for why the scalar Ising/Langevin
LCL experiments may fail even when the paper's XY LCL model learns.

## Capacity Is Probably Not The Main Difference

The important distinction is not that the XY model obviously has more useful
classification capacity. A scalar Ising-like network with local couplings should
also be able to represent learnable MNIST features in principle, especially for a
simple task like MNIST.

The more relevant distinction is the response to an infinitesimal output nudge.
Equilibrium propagation depends on the nudged phase producing a meaningful local
differential response around the free equilibrium. In an XY model, each state is
an angle. A small nudging force can rotate a phase continuously, so nearby
equilibria can often be connected by a smooth angular deformation.

In a scalar Ising-like system, even with continuous Langevin states in `[-1, 1]`,
the effective energy landscape can still behave like a rugged basin structure.
The local minimum selected by the free phase does not necessarily transform
smoothly under a small target perturbation. A small beta can therefore produce
almost no useful contrastive signal, while a large beta can push the system into
a different basin and create a biased or unstable update.

## Why This Matters For The LCL Runs

The paper's LCL result uses:

- an XY phase model,
- cosine pair interactions,
- angular output targets,
- beta `0.1`,
- learning rate `1e-4`,
- `Minit = 1` for MNIST.

Our current LCL implementation reproduces the local 6x6 topology and manager
training path, but it uses scalar hidden/output states, bilinear couplings,
magnetic fields, and scalar output nudges. This is a useful experiment, but it is
not a faithful test of the paper's continuous angular response mechanism.

The low-beta scalar failures should therefore be interpreted as evidence that
the scalar model is not yet getting a reliable differential contrastive response,
not as evidence that local Ising-style networks cannot learn.

## What To Test Next

The next fixes should target the response problem directly:

- measure the free-to-nudged output and hidden response as a function of beta;
- check whether small beta gives a nonzero, class-aligned signal or only noise;
- test averaging over multiple initial states when the selected basin is unstable;
- test time-averaged state estimates after burn-in to reduce stochastic readout noise;
- test schedules that keep the free phase exploratory but make the nudged response more local;
- implement a faithful XY/cosine diagnostic to separate topology issues from state-geometry issues.

The practical goal is not to copy XY because it has more capacity. The goal is to
recover the useful property that the target perturbation produces a smooth,
informative change in the sampled state.
