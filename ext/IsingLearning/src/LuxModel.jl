export LayeredIsingGraphLayer, ep_train_step!, edge_term_from_state, ep_edge_derivative_estimate

"""
    edge_term_from_state(state_vec, i, j) -> Float32

Return the pairwise spin product `s[i] * s[j]` for one state vector.

This is the per-edge correlation term used inside EP updates.
"""
function edge_term_from_state(state_vec::AbstractVector, i::Integer, j::Integer)
    return Float32(state_vec[i]) * Float32(state_vec[j])
end

"""
    ep_edge_derivative_estimate(i, j, s_nudged_plus, s_nudged_minus, beta) -> Float32

Mock one-sample EP derivative estimate for edge `(i, j)`:

`(s_nudged_plus[i] * s_nudged_plus[j] - s_nudged_minus[i] * s_nudged_minus[j]) / beta`
"""
function ep_edge_derivative_estimate(
    i::Integer,
    j::Integer,
    s_nudged_plus::AbstractVector,
    s_nudged_minus::AbstractVector,
    beta::Real,
)

    beta_f32 = Float32(beta)
    iszero(beta_f32) && throw(ArgumentError("beta must be non-zero"))

    free_term = edge_term_from_state(s_nudged_minus, i, j)
    nudged_term = edge_term_from_state(s_nudged_plus, i, j)
    return (nudged_term - free_term) / beta_f32
end

# =====================================================================
#  LayeredIsingGraphLayer — Lux layer for Equilibrium Propagation on IsingGraphs
# =====================================================================

"""
    LayeredIsingGraphLayer(graph_init; input_idxs, output_idxs, β = 0.1f0)

Lux layer wrapping an `InteractiveIsing.IsingGraph` for
Equilibrium Propagation (EP).

## Architecture (immutable — no arrays stored in the struct)
- `graph_init`:  zero-arg function `() -> IsingGraph(...)` for lazy construction
- `input_idxs`:  graph-level spin indices that receive clamped input
- `output_idxs`: graph-level spin indices read as the layer's output
- `β`:           default nudging strength for the clamped phase

## Learnable parameters (`ps`)
- `weights`: `Vector` — adjacency nonzero values (coupling strengths)
- `biases`:  `Vector` — per-spin bias / magnetic field

## Managed state (`st`)
- `graph`: the live, mutable `IsingGraph` (simulation happens in-place)

## Forward pass
`(layer)(x, ps, st) → (y, st)` executes the **free phase**:
1. Sync `ps` back into the graph (`weights → adj`, `biases → :b`)
2. Clamp input spins to `x`
3. Run the simulation process (free phase relaxation)
4. Read output spins → `y`

The **nudged phase** and **weight update** live in [`ep_train_step!`](@ref),
which is called from your training loop.
"""
struct LayeredIsingGraphLayer{G,I,O} <: LuxCore.AbstractLuxLayer
    model_graph::G
    input_layer::I
    output_layer::O
    β::Float32
    fullsweeps::Int
    nunits::Int
end

function LayeredIsingGraphLayer(graph_init;
                      input_idxs,
                      output_idxs,
                      β::Real = 0.1f0,
                      fullsweeps::Integer = 10)

    graph_init = graph_init isa Function ? graph_init() : graph_init
    n_units = nstates(graph_init)
    LayeredIsingGraphLayer(graph_init,
                 input_idxs,
                 output_idxs,
                 Float32(β),
                 fullsweeps,
                 n_units)

end

# ─────────────────────────────────────────────────────────────────────
#  Lux interface
# ─────────────────────────────────────────────────────────────────────

function initialparameters(rng::AbstractRNG, layer::LayeredIsingGraphLayer)
    g = layer.model_graph  # throwaway graph just to read the initial values
    return (
        w = deepcopy((SparseArrays.getnzval(adj(g)))),
        b  = rand(rng, eltype(g), nstates(g)),  # placeholder random biases
        α_i     = rand(rng, eltype(g), nstates(g))  # placeholder random self-energies
    )
end

function initialstates(rng::AbstractRNG, layer::LayeredIsingGraphLayer)
    return (
        graph = layer.model_graph  # the live graph state will be managed inside `st` and mutated in-place during the forward pass,
    )
end

# ─────────────────────────────────────────────────────────────────────
#  Parameter sync: push Lux `ps` into the mutable graph
# ─────────────────────────────────────────────────────────────────────

"""
    sync_params!(g, ps)

Write the learnable parameters from the Lux `ps` NamedTuple
back into the graph `g` before running a simulation phase.
"""

function sync_params!(g::IsingGraph, ps)
    nonzeros(offdiag(adj(g))) .= ps.w
    biases = getparam(g.hamiltonian, Magfield, :b)
    biases .= ps.b
    self_energies = diag(adj(g))
    self_energies .= ps.α
    return g
end


# ─────────────────────────────────────────────────────────────────────
#  Forward pass  (free phase)
# ─────────────────────────────────────────────────────────────────────

function (layer::LayeredIsingGraphLayer)(x, ps, st)
    g = st.graph

    # 1. Push learnable weights / biases into the graph
    # sync_params!(g, ps)

    # 2. Clamp input spins to x
    off!(g.index_set, layer.input_layer)  # ensure the input layer is turned off
    state(g[1]) .= x  # TODO: this assumes the input layer is layer 1; generalize to arbitrary input_layer

    # 3. Run free phase (Ising hamiltonian only, clamping β = 0)
    algo = st.relax_routine
    #TODO: Use named variant when available
    # For not algo[1] is the dynamics 
    run(algo, Input(algo[1], state = g), lifetime = Repeat(length(state(g))*layer.fullsweeps), mode = :inline_synced)

    # 4. Read output spins
    y = copy(state(g[nlayers(g)]))  # TODO: this assumes the output layer is the last layer; generalize to arbitrary output_layer
    
    return y, st  # st keys unchanged → Lux is happy
end

# =====================================================================
#  EPClamping — ProcessAlgorithm for the nudged phase
# =====================================================================

"""
    EPClamping(β, target) <: ProcessAlgorithm

ProcessAlgorithm that activates the `Clamping` Hamiltonian on the graph.
Compose with your MC algorithm (e.g. `Metropolis`) inside a
`CompositeAlgorithm` so the simulation loop alternates between
MC updates and clamping-parameter management.

Example (pseudocode):
```julia
ep = EPClamping(0.1f0, target_vector)
algo = CompositeAlgorithm(Metropolis(), ep, (N_mc, 1),
                          Route(ep => Metropolis, ...))
createProcess(g, dynamics = algo, lifetime = total_steps)
wait(g)
```
"""
struct EPClamping <: ProcessAlgorithm
    β::Float32
    target::Vector{Float32}
end

function Processes.init(alg::EPClamping, context)
    # TODO: ensure the graph uses CompositeHamiltonian(Ising, Clamping)
    #       activate the :β and :y parameters via setparam!
    return (;)
end

function Processes.step!(alg::EPClamping, context)
    # TODO: write alg.β and alg.target into the graph's
    #       Clamping Hamiltonian parameters each time this fires
    # setparam!(g, :β, alg.β)
    # setparam!(g, :y, alg.target, true, si, ei)  # for the output layer
    return (;)
end

# =====================================================================
#  EP training step  (called from your training loop, not from Lux AD)
# =====================================================================

"""
    ep_train_step!(layer, x, target, ps, st; β, η) → (ps_new, st)

One full Equilibrium Propagation parameter update:

1. **Free phase** — clamp inputs, relax under Ising Hamiltonian (β_clamp = 0),
   record output correlations  ⟨sᵢ sⱼ⟩_free.
2. **Nudged phase** — activate Clamping Hamiltonian (β_clamp = `β`, y = `target`),
   relax again, record  ⟨sᵢ sⱼ⟩_nudged.
3. **Weight update** —
   ``Δw_{ij} = \\frac{1}{β}\\left(⟨s_i s_j⟩_{\\text{nudged}} - ⟨s_i s_j⟩_{\\text{free}}\\right)``
   applied as  `ps.weights .+= η .* Δw`

Returns a **new** `ps` NamedTuple (Lux convention: don't mutate ps).
"""
function ep_train_step!(layer::LayeredIsingGraphLayer, x, target, ps, st;
                        β::Real  = layer.β,
                        η::Real  = 1f-3)
    g = st.graph
    sync_params!(g, ps)

    # ── Free phase ─────────────────────────────────────────────
    # TODO: clamp inputs, deactivate clamping, run process, wait
    # s_free = copy(state(g))

    # ── Nudged phase ───────────────────────────────────────────
    # TODO: activate clamping β & target, run process, wait
    # s_nudged = copy(state(g))

    # ── Compute correlations & weight update ───────────────────
    # TODO: for each edge (i,j) in adjacency:
    #   Δw_ij = (1/β) * (s_nudged[i]*s_nudged[j] - s_free[i]*s_free[j])
    # new_weights = ps.weights .+ η .* Δw
    # Similarly for biases:
    #   Δb_i  = (1/β) * (s_nudged[i] - s_free[i])

    # Return a NEW NamedTuple (Lux immutability convention)
    # ps_new = (weights = new_weights, biases = new_biases)
    ps_new = ps  # placeholder until the above is filled in

    return ps_new, st
end
