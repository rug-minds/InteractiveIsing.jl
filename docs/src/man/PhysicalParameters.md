# Physical Parameters

InteractiveIsing can accept `Unitful` quantities at construction and explicit
setter boundaries while keeping Hamiltonians unit-free during simulation. The
conversion happens once, before the usual parameter `ensure` logic stores
plain `Float32` or `Float64` values.

This means the hot `calculate`, `DeltaH`, and `update!` paths stay type-stable:
physical units are metadata for inputs and outputs, not runtime Hamiltonian
fields.

## Scale Context

Use `PhysicalScales` to define the physical meaning of one internal unit:

```julia
using InteractiveIsing
using Unitful

scales = PhysicalScales(
    energy = 1u"meV",
    length = 1u"nm",
    temperature = 1u"meV",
)
```

The available scale roles are:

- `energy`
- `length`
- `temperature`
- `charge`
- `dipole`
- `state`

State values are dimensionless by default. Only set a state scale when a term
explicitly needs a physical state unit.

For Monte Carlo temperature, an explicit `temperature` scale is used when set.
If it is not set and the input temperature has energy units, the `energy` scale
is used instead. This is the common `k_B T` convention.

Attach scales through the graph constructor:

```julia
g = IsingGraph(
    32,
    32,
    Continuous(),
    Ising(b = 0.25u"meV");
    physical_scales = scales,
    temperature = 2.15u"meV",
    precision = Float32,
)
```

Here `ExtField.b` is stored internally as `0.25`, because the energy scale is
`1u"meV"`. The graph temperature is stored internally as `2.15`.

!!! note
    InteractiveIsing does not insert physical constants during conversion. If
    an algorithm interprets `temp(g)` as an energy scale, pass `k_B T` in
    energy units, for example `2.15u"meV"` for roughly room-scale tens of
    kelvin. Use a `temperature = 1u"K"` scale only if your algorithm and model
    intentionally use kelvin numerically.

## Hamiltonian Parameters

Template-based Hamiltonian parameters carry physical metadata in
`ParameterSpec.units`. Plain numeric values pass through unchanged. Unitful
values are converted by `internalvalue(value, spec, scales, model)`.

Common built-in defaults are:

| Term | Parameter | Default physical role |
| --- | --- | --- |
| `Bilinear` | `J` | energy |
| `ExtField` | `b` | energy |
| `PolynomialHamiltonian` | `c` | energy |
| `CosineInteraction` | `J` | energy |
| `CosineInteraction` | `phase`, `edge_phase` | dimensionless angle |
| `Clamping` | `beta` | energy |
| `Clamping` | `y` | state |
| `SoftplusMarginNudging` | `beta` | energy |
| `SoftplusMarginNudging` | `y`, `tau` | state |
| `DepolField` | `c` | energy |
| `CoulombHamiltonian` | `scaling` | dipole |
| `CoulombHamiltonian` | screening lengths | length |
| `GaussianBernoulli` | ML parameters | dimensionless |

Ambiguous or custom terms should set metadata explicitly with
`physicalunits(...)`. For example, a coupling with units of energy per length
would use:

```julia
physicalunits(energy = 1, length = -1, role = :line_tension)
```

If a Unitful input reaches a parameter without metadata, conversion throws
`MissingPhysicalRole`. If metadata names a scale that is not available,
conversion throws `MissingPhysicalScale`. Both errors include the parameter name
and the missing scale role.

## Setters And Reading Back Values

Use `setphysical!` when writing Unitful values into an already instantiated
Hamiltonian:

```julia
setphysical!(g, InteractiveIsing.ExtField, :b, 0.4u"meV")
```

This updates the existing mutable parameter storage with the converted internal
value. Use `physicalvalue` to read the stored value back in physical units:

```julia
b = physicalvalue(g, InteractiveIsing.ExtField, :b)
```

The low-level Hamiltonian storage remains unit-free:

```julia
mag = InteractiveIsing.gethamiltonian(g.hamiltonian, InteractiveIsing.ExtField)
mag.b[1] # plain numeric internal value
```

Graph temperature also accepts Unitful values. Energy-unit values use
`PhysicalScales.temperature` if it is set, otherwise they fall back to
`PhysicalScales.energy`:

```julia
temp!(g, 3.0u"meV")
```

In the Windows interface, the temperature slider still edits the internal
numeric value. Its label is display-only scale-aware: it uses
`PhysicalScales.temperature` when available, otherwise it falls back to
`PhysicalScales.energy`. If the selected display scale has energy units, it is
interpreted as `k_B T` and shown as the corresponding kelvin value.

## Topology Lengths

Topology constructors and `LatticeConstants` accept Unitful lengths:

```julia
g = IsingGraph(
    32,
    32,
    Continuous(),
    LatticeConstants(0.5u"nm", 0.5u"nm");
    physical_scales = PhysicalScales(energy = 1u"meV", temperature = 1u"meV"),
)
```

If `physical_scales.length` is already set, topology inputs are converted
against that scale. If no length scale is set but topology inputs are Unitful,
the first Unitful length's unit is inferred and written into the shared graph
scale context. In the example above, internal lattice constants are stored as
`0.5`, and `physicalscales(g).length[] == 1u"nm"`.

For standalone topologies, pass a scale explicitly when you want the same
behavior outside a graph constructor:

```julia
scales = PhysicalScales(length = 1u"nm")
top = SquareTopology((32, 32); lattice_constants = (0.5u"nm", 0.5u"nm"), physical_scales = scales)
```

## Physical Weight Generators

Wrap a normal weight generator in `PhysicalWeightGenerator` when the generator
function should receive Unitful distances or return Unitful weights:

```julia
function exchange_weight(; dr, c1 = nothing, c2 = nothing, dc = nothing)
    x = Unitful.ustrip(Unitful.NoUnits, dr / (0.35u"nm"))
    return dr <= 0.55u"nm" ? 8u"meV" * exp(-x) : 0u"meV"
end

wg = PhysicalWeightGenerator(WeightGenerator(exchange_weight, 1))
```

The sparse adjacency builder still works with internal numeric distances. At
the edge-weight boundary, `getWeight` reconstructs `dr` using
`physicalscales(g).length[]`, calls the wrapped generator, and converts any
Unitful returned weight into internal `J` units. The default returned-weight
metadata is `physicalunits(energy = 1)`.

Use the optional `units` keyword when a generator returns a different physical
role:

```julia
wg = PhysicalWeightGenerator(
    WeightGenerator(my_weight, 1);
    units = physicalunits(energy = 1, length = -2, role = :surface_coupling),
)
```

## Example

See `examples/Physical Parameters.jl` for a complete 3D non-GUI example using
meV energies, nm lattice spacing, Unitful Hamiltonian inputs, kelvin
temperatures converted explicitly to `k_B T`, and a physical nearest-neighbor
weight generator.

See `examples/Physical Coulomb Parameters.jl` for the same physical-scale
pattern with `CoulombHamiltonian`, Unitful dipole scaling, and Unitful screening
lengths.
