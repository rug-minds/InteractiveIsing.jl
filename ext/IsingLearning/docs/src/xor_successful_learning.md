# XOR Successful Learning Run

This note records the first XOR setup in this branch that clearly learned. The
matching runnable snapshot is:

```text
ext/IsingLearning/examples/xor_succesful_learning.jl
```

The filename uses the spelling requested in the implementation task.

## Exact Run

The successful probe was:

```text
ISING_XOR_STAT_EPOCHS=1000
ISING_XOR_STAT_LOG_EVERY=100
ISING_XOR_STAT_MINIT=4
ISING_XOR_STAT_EVAL_REPEATS=16
ISING_XOR_STAT_FREE_RELAXATION=50
ISING_XOR_STAT_NUDGED_RELAXATION=50
ISING_XOR_STAT_TEMP=0.001
ISING_XOR_STAT_DIR=ext/IsingLearning/runs/xor_statistical_ep_probe_1000_cold
julia --project=ext/IsingLearning ext/IsingLearning/examples/xor_statistical_ep.jl
```

`xor_succesful_learning.jl` has the same defaults, under the
`ISING_XOR_SUCCESS_*` environment variable prefix.

## Result

The run moved from a near-random baseline to a solved XOR classifier:

```text
epoch 0:     MSE = 1.002244, accuracy = 0.5
epoch 200:   MSE = 0.666538, accuracy = 1.0
epoch 500:   MSE = 0.184951, accuracy = 1.0
epoch 900:   MSE = 0.095809, accuracy = 1.0
epoch 1000:  MSE = 0.105719, accuracy = 1.0
restored:    MSE = 0.088773, accuracy = 1.0
```

Restored mean outputs:

```text
(false, false) target [ 1, -1]  mean [ 0.6997, -0.6884]
(false, true)  target [-1,  1]  mean [-0.7833,  0.7766]
(true, false)  target [-1,  1]  mean [-0.5767,  0.5766]
(true, true)   target [ 1, -1]  mean [ 0.8615, -0.7800]
```

Artifacts:

```text
ext/IsingLearning/runs/xor_statistical_ep_probe_1000_cold/
```

## What Made It Work

The important changes were structural, not just more epochs.

1. **Simple all-to-all task encoding**

   Input is four bipolar one-hot units, one per XOR case:

   ```text
   active case = +1
   inactive cases = -1
   ```

   Output is two bipolar one-hot units:

   ```text
   XOR false = [ 1, -1]
   XOR true  = [-1,  1]
   ```

   This removed the global-pattern symmetry/readout complications from the
   earlier experiments. Those complications matter later for local-connectivity
   experiments, but they were noise for the first all-to-all learning test.

2. **Direct output clamping**

   Because the output target is now directly a two-spin target, the standard
   `Clamping` Hamiltonian is exactly the intended squared error on output
   spins:

   ```math
   C(s, y) = \frac{1}{2}\sum_i (s_i - y_i)^2.
   ```

   The previous scalar readout setup needed custom readout clamping, and that
   made it harder to tell whether the architecture, readout map, or learning
   rule was failing.

3. **Repeated-state statistical gradient**

   Each training epoch averages contrastive gradients over repeated initial
   states:

   ```text
   4 XOR samples * Minit=4 = 16 trajectories per epoch
   ```

   This matters because the system is stochastic and multistable. A single
   relaxed state is a poor estimate of the model's behavior. The averaged
   signal is less sensitive to one unlucky basin.

4. **Cold but finite temperature**

   The working temperature was:

   ```text
   T = 0.001
   ```

   `T=0.05` produced a much noisier output distribution and did not show a
   convincing trend in the 100-epoch probe. `T=0.001` kept enough stochasticity
   to sample different starts while making the output means trainable.

5. **No local potential / no double well**

   The Hamiltonian is:

   ```text
   Bilinear + MagField + Clamping
   ```

   There is no local-potential term and no double-well term. Bounded states are
   already enforced by the state set `[-1, 1]`.

6. **Trainable magnetic-field bias**

   There is no forced constant input unit. The graph already has trainable
   `MagField` biases. These biases are enough for the all-to-all one-hot XOR
   setup.

7. **Fixed validation seeds**

   Validation uses a fixed set of repeated seeds across logged epochs. This made
   the curve interpretable. Earlier runs sometimes selected a "best" point
   against moving validation noise.

8. **Best-epoch restoration**

   The final epoch was not the best epoch. The run restored the best validation
   MSE point, which was epoch 900 in the CSV. That restored point had MSE about
   `0.089`.

## Phase Initialization

The clamped phases do not start from a fresh random state.

For each trajectory, the sequence is:

```text
reset/random state
-> apply input
-> free relaxation
-> copy free relaxed state into equilibrium_state
-> restore equilibrium_state, apply input and target, set +β
-> plus/nudged relaxation
-> restore the same equilibrium_state, apply input and target, set -β
-> minus/anti-nudged relaxation
-> contrastive gradient from plus minus minus
```

In code this is handled by `Forward_and_Nudged` in
`ext/IsingLearning/src/Dynamics.jl`.

- `ForwardDynamics` calls `initstate!`, applies the input, relaxes, and stores
  the free state in `equilibrium_state`.
- `NudgedDynamics.plus` calls `setgraph!(..., target = equilibrium_state)`,
  reapplies the input/target, sets `+β`, then relaxes.
- `NudgedDynamics.minus` restores the same `equilibrium_state`, reapplies the
  input/target, sets `-β`, then relaxes.

This means `FREE_RELAXATION=50` is the number of steps used to obtain the free
state, and `NUDGED_RELAXATION=50` is the number of additional steps after
starting from that free state under the clamped Hamiltonian.

## Why 50 Relaxation Steps Was Enough Here

This is not the same relaxation problem as the old 1500-step runs.

The earlier setup tried to approximate a near-equilibrium state of a hard,
bounded, pattern/readout system. We were asking a single trajectory to settle
well enough that a small clamping perturbation gave a meaningful EP signal. That
made relaxation length critical, and 100-ish steps was visibly far from settled.

The successful setup is different:

- the task encoding is much simpler;
- the output target is directly clamped, not passed through a scalar readout;
- the clamped phases start from the already free-relaxed state;
- the training signal is averaged over several stochastic starts;
- we only need a useful biased contrast per trajectory, not a fully converged
  deterministic fixed point;
- `BlockLangevin(adjusted=false, stepsize=0.05)` moves the small active layer
  quickly enough for this simplified task.

So `50` free and `50` nudged relaxation calls were enough to produce a useful
statistical gradient. This should not be interpreted as a universal EP
relaxation budget. For harder encodings, local connectivity, lower noise, or
near-deterministic fixed-point claims, relaxation has to be remeasured.

## Remaining Caveat

The output means classify all four cases correctly, and MSE is below `0.1`, but
the output standard deviations are still noticeable. The next robustness check
should evaluate the restored parameters over more validation repeats and several
training seeds.
