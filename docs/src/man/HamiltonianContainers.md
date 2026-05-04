# Hamiltonian Containers

Hamiltonian parameters can be ordinary arrays, but many useful terms are
spatially uniform or only need a small mutable value. The package provides
array-like containers for those cases. They are especially useful because the
Hamiltonian code can use one interface, `x[]` or `x[i]`, while the container
decides whether that access is a real memory load, a shared mutable value, or a
compile-time constant.

## `ConstVal`

`ConstVal(x)` is a zero-dimensional constant container. It is useful for scalar
couplings:

```julia
Ising(c = ConstVal(0f0))
Quartic(c = ConstVal(1f0))
```

Use it when the value is fixed for the lifetime of the simulation.

## `ConstFill`

`ConstFill(x, n...)` is an `AbstractArray` where every index returns the same
constant. The value is stored in the type, so after specialization the compiler
can often constant-fold the access.

```julia
Ising(b = ConstFill(0f0))
```

For Hamiltonian calculations this matters a lot. If a term multiplies by a
`ConstFill(0)` or a `ConstVal(0)`, the hot calculation path can become almost
equivalent to not having that contribution at all: the indexed load is a
compile-time constant, arithmetic with zero can be eliminated, and the
remaining code is drastically reduced. The term object may still exist and the
outer dispatch/bookkeeping may still happen, but the actual calculation can be
compiled down to essentially nothing for that contribution.

Use `ConstFill` for fixed uniform fields, masks, local potentials, or defaults
where allocating a full vector would only waste memory and bandwidth.

## `UniformArray`

`UniformArray(x, n...)` is also uniform, but mutable. Every index reads the
same `Ref` value and writes update that shared value.

```julia
Ising(b = UniformArray)
```

This is useful when a parameter is spatially uniform but changes during a run,
for example a global field sweep. It avoids storing a full vector while still
allowing updates.

## `OffsetArray`

`OffsetArray(vec, offset)` stores a real backing array plus an additive offset.
Reads return `offset + vec[i]`, and writes store `value - offset` into the
backing array.

```julia
b = OffsetArray(zeros(Float32, n), 0.2f0)
Ising(b = b)
```

Use it when a parameter has a global baseline plus local deviations. This keeps
the global part separate from the local memory and can be more ergonomic than
manually adding offsets in every Hamiltonian.

## Passing Containers To Hamiltonians

For state-like parameters such as `b`, `localpotential`, or `y`, you can pass:

- a scalar value, e.g. `b = 1`;
- a singleton vector, e.g. `b = [1]`, which is expanded to graph length;
- a container value, e.g. `b = ConstFill(1f0)`;
- a container type, e.g. `b = UniformArray`, which asks the template to build
  the container with the term default value and graph size;
- a custom vector or matrix;
- a graph function, e.g. `localpotential = g -> adj(g).diag`.

Examples:

```julia
Ising(b = 1)
Ising(b = [1])
Ising(b = ConstFill(1f0))
Ising(b = UniformArray)
Ising(b = OffsetArray)
Quartic(localpotential = g -> adj(g).diag)
```

For scalar-like parameters such as `c` or `β`, use:

```julia
Quadratic(c = ConstVal(0f0))
Quartic(c = UniformArray(1f0))
Clamping(UniformArray(1f0))
```

The term template will convert plain numbers to the graph precision by default.
Use `NoEnsure(x)` when the storage should be accepted as-is but still checked,
and `Force(x)` when the template should not check it.

If a keyword is omitted, the term uses its optimized default storage. For
example, omitted `MagField.b` uses a constant zero field. If a scalar or
singleton is explicitly passed, the term uses its explicit-input storage
policy. For `MagField.b`, that means `b = 1` and `b = [1]` become mutable
uniform graph-sized storage.

## Choosing A Container

Use `ConstVal` for fixed scalar coefficients.

Use `ConstFill` for fixed uniform arrays. This is the best choice when the
value is zero or one and you want the compiler to remove the calculation as far
as possible.

Use `UniformArray` for uniform values that change during the run.

Use `OffsetArray` for a global offset plus local deviations.

Use `Vector` or `Array` when every site really has independent memory.

## Container Auto-Fill Interface

Any array type that should be usable as an auto-filled Hamiltonian storage type
must implement `filltype`:

```julia
filltype(::Type{MyArray}, value, dims...) = MyArray(value, dims...)
```

The Hamiltonian template calls this when a user passes a storage type such as
`b = Vector` or when a scalar must be expanded according to a term's
`default_type`. If no `filltype` method exists, construction fails with a clear
error instead of silently falling back to dense `fill`.
