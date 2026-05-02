# Indexing

Graphs carry an `index_set` object that decides which spins are available to update.
The graph accessor and the sweep accessor are intentionally separate.

`index_set(graph)` returns the stored index-set object itself.
Use this when you need to toggle layers, inspect defects, or pass the picker into
proposal code.

`pick_idx(rng, index_set)` is the single-site API.
Algorithms and proposers that update one spin at a time by drawing a random site
use `pick_idx`. This is the only operation needed by `Metropolis`, `KineticMC`,
and the standard proposers.

`sampling_indices(index_set)` is the sweep API.
Algorithms that want to visit each currently active site once per sweep use
`sampling_indices` to materialize the active indices. `LocalLangevin` uses
this path.

`sampling_indices(graph)` is a convenience wrapper for
`sampling_indices(index_set(graph))`.

Built-in index-set shapes:

- Plain ranges, vectors, and sets already work as both pickers and sweep inputs.
- `GraphDefectsNew` exposes its active sites through `aliveindices`.
- `ToggledIndexSet` stores one range per layer and can toggle layers on or off.
- `ToggledLayerIndexSet` switches between two precomputed index collections.

If you implement a custom index set, the minimum contract depends on how you want
to use it:

- For proposer-based random updates, implement `InteractiveIsing.pick_idx(rng, your_index_set)`.
- For sweep-based algorithms, also implement `InteractiveIsing.sampling_indices(your_index_set)`.

Minimal example:

```julia
struct MyIndexSet <: InteractiveIsing.UniformIndexPicker
	active::Vector{Int}
end

InteractiveIsing.pick_idx(rng, is::MyIndexSet) = rand(rng, is.active)
InteractiveIsing.sampling_indices(is::MyIndexSet) = is.active
```

Using a custom index set in a graph constructor:

```julia
g = IsingGraph(
	Layer(4, Continuous(), Coords(0, 1, 0)),
	Layer(4, Continuous(), Coords(0, 2, 0));
	index_set = graph -> ToggledIndexSet(graph),
)
```

The important rule is that `pick_idx` and `sampling_indices` should describe the
same currently active subset of spins. If they disagree, random-site algorithms
and sweep-based algorithms will evolve different systems.
