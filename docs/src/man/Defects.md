# Mobile Defects and Charges

`DefectHopping` is the single-species graph-attached proposer used for neutral
mobile local-potential defects. `DefectsModel` is the two-species add-on Monte
Carlo model for vacancies and mobile carriers coupled to an `IsingGraph`.

For a single neutral species, attach the coupling hamiltonian directly:

```julia
vacancy_hamiltonian =
    ExternalFieldShiftCoupling(0.06; hopping_scale = 100) +
    LocalPotentialShiftCoupling(2, 0.03; hopping_scale = 100)

defects = DefectHopping(g; defects = [CartesianIndex(3, 3, 2)], hamiltonian = vacancy_hamiltonian)
```

## Model Shape

`DefectsModel` follows the same model/proposer split as `IsingGraph`:

```julia
vacancy_hamiltonian =
    CoulombChargeCoupling(2q; split = 0.5) +
    ExternalFieldChargeCoupling() +
    LocalPotentialShiftCoupling(2, 0.012)

electron_hamiltonian =
    CoulombChargeCoupling(-q; split = 0.5) +
    ExternalFieldChargeCoupling()

charges = DefectsModel(
    g;
    vacancies = MobileVacancies(10; charge = 2q, hamiltonian = vacancy_hamiltonian),
    charges = MobileCharges(20; charge = -q, hamiltonian = electron_hamiltonian),
    electron_attempt_rate = 10,
)

context = StatefulAlgorithms.init(Metropolis(), (; model = charges))
```

`Metropolis` calls `get_proposer(charges)` internally. The derived
`ChargeHopProposer` draws `ChargeHopProposal`s with `rand`, while the model owns
the vacancy/carrier state and exposes `state(charges)`, `graphstate(charges)`,
`temp(charges)`, and `charges.hamiltonian`.

`DefectsModel <: AddOnAbstractMonteCarloModel`, so it declares
`requires(DefectsModel) == (AbstractIsingGraph,)` and `dependson(charges)`
returns the concrete graph dependency.

## Species Entries

Use `MobileVacancies` for the positive vacancy-like species and `MobileCharges`
for the mobile carrier species.

```julia
MobileVacancies(10; charge = 2q, hamiltonian = vacancy_hamiltonian)
MobileCharges([CartesianIndex(2, 3, 1)]; charge = -q, hamiltonian = electron_hamiltonian)
```

The first argument is either:

- an integer count, which requests random initialization on the bound graph
  layer;
- an explicit collection of local indices, tuples, or `CartesianIndex` values
  for manual initialization.

Random carrier initialization avoids already chosen vacancy sites. Manual
initialization preserves the supplied positions and still validates duplicate
positions within each species.

## Coupling Hamiltonians

Species hamiltonians use the same `+` syntax as graph hamiltonians. These terms
are coupling hamiltonians: they define how a mobile species perturbs graph
Hamiltonian storage and how a proposed hop contributes to `ΔH`.

- `CoulombChargeCoupling(charge; split = 0.5)` registers mobile free-charge
  occupancy in `CoulombHamiltonian`. The magnitude must match the corresponding
  `q_positive` or `q_negative` in the Coulomb term.
- `ExternalFieldChargeCoupling(; axis = nothing, hopping_scale = 1)` couples a
  charged hop to the graph `ExtField`. `axis = nothing` uses the proposal's last
  coordinate axis.
- `LocalPotentialShiftCoupling(order, strength; hopping_scale = 1)` adds
  `strength` to mutable `PolynomialHamiltonian{order}` local-potential storage
  at occupied sites.
- `LocalPotentialScaleCoupling(order, factor; hopping_scale = 1)` multiplies
  mutable `PolynomialHamiltonian{order}` local-potential storage at occupied
  sites by `factor`. Accepted hops divide the old site by `factor` and multiply
  the new site by `factor`.
- `ExternalFieldShiftCoupling(strength; hopping_scale = 1)` adds `strength` to
  mutable `ExtField.b` storage at occupied sites.

The older names `CoulombChargeShift`, `LocalPotentialShift`, `ExtFieldShift`,
and `ExtFieldChargeCoupling` remain as aliases.

## Relative Attempt Rates

Pass only one relative rate. The omitted species uses rate `1`.

```julia
DefectsModel(g; vacancies, charges, electron_attempt_rate = 10)
DefectsModel(g; vacancies, charges, vacancy_attempt_rate = 0.1)
```

Both examples make vacancies one tenth as likely per particle as carriers. Passing
both vacancy and carrier rates is rejected so the normalization is unambiguous.

The proposal probability is proportional to:

```text
species_attempt_rate * number_of_particles_in_species
```

## Compatibility

`ChargeHopProposer(g; ...)` is a compatibility constructor that now returns a
`DefectsModel` model. New code should call `DefectsModel` directly.
`MobileChargeHopping` remains an alias for `DefectsModel`.
The concrete `ChargeHopProposer(model)` remains the derived proposal source used
by `Metropolis`.
