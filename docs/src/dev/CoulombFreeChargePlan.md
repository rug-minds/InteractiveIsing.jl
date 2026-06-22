# Coulomb Free Charge Notes

This note records how mobile charged defects and carrier-like charges fit into
the current `CoulombHamiltonian` implementation.

## Current Solver Shape

`CoulombHamiltonian` solves from the total sheet charge field `ρ`. The solver
does not care whether charge came from dipoles or from mobile charges:

```text
ρ_total = ρ_bound_from_dipoles + ρ_free
```

The bound-charge assumption is in `init!`, where each dipole contributes a
negative lower sheet charge and positive upper sheet charge. `recalc!` simply
solves from the current total `ρ`.

## Free-Charge State

Mobile free charge is stored separately from `ρ` as conserved vacancy and
carrier occupancy fields. `DefectsModel` groups the two fields into one
Monte Carlo model, so `state(charges)` returns a charge-state object with
vacancy/carrier indices and occupancies, while `graphstate(charges)` still
returns the coupled spin state. `get_proposer(charges)` derives the
`ChargeHopProposer` that draws `ChargeHopProposal`s for `Metropolis`.

The Coulomb term stores:

- positive and negative cell occupancies on the dipole grid,
- positive and negative sheet occupancies on the Coulomb sheet grid,
- configured charge magnitudes `q_positive` and `q_negative`.

The total sheet charge `ρ` is derived by rebuilding from the current dipoles and
these free-charge occupancies. A positive cell charge contributes `+/+` to the
two neighboring sheets; a negative cell charge contributes `-/-`. Direct sheet
occupancy contributes to the sheet where it lives.

## Neutrality

The free-charge mode requires explicit neutrality:

```text
q_positive * N_positive == q_negative * N_negative
```

This keeps the existing zero-mode handling as a gauge convention. There is no
hidden uniform neutralizing background. A charged vacancy simulation should add
explicit negative carrier occupancy, electrode/sheet charge, or another modeled
countercharge.

## Defect Coupling

`CoulombChargeCoupling(q)` registers a mobile free-charge occupancy:

- positive `q` uses positive occupancy,
- negative `q` uses negative occupancy,
- `abs(q)` must match the corresponding `CoulombHamiltonian` charge magnitude.

Oxygen vacancies can additionally carry structural modes such as
`LocalPotentialShiftCoupling`, `LocalPotentialScaleCoupling`, or
`ExternalFieldShiftCoupling`. Electron-like negative carriers should normally use
only `CoulombChargeCoupling(-q)` unless a separate physical model says they also
perturb local structure.

`interface(charges)` displays both fields in one window, with positive and
negative charges rendered in different colors.

## Energy Path

The current implementation keeps the exact path for correctness:

1. Move the proposed occupancy temporarily.
2. Rebuild `ρ`.
3. Recalculate the Coulomb potential.
4. Compare `0.5 * sum(ρ .* u)` before and after.
5. Restore buffers for rejected proposals.

Accepted moves update occupancy, rebuild `ρ`, and recalculate `u`. A faster
incremental Fourier update can be added later against this same occupancy API.
