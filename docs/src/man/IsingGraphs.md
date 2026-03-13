# IsingGraphs

The `IsingGraphs{T <: AbstractFloat}` are the main datastructure used for this module.

The main datatypes in the IsingGraph relevant for runtime computations as the state and the adjacency list. They may be accessed through the accessor functions `state(g)` and `adj(g)` respectively. The subtype `T` is a `Float` and gives the precision for the state and weights.

It also holds the temperature, but this may be moved to the layers.

`state` returns a `Vector{T}` and `adj` a `SparseMatrixCSC{T,Int32}`.

## The State

## The Adjacency List

## Setting the Hamiltonian

## Setting the algorithm