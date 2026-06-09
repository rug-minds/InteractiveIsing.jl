using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "..", ".."))

using LinearAlgebra
using Random
using SparseArrays

include(joinpath(@__DIR__, "..", "mnist_lcl_6x6_window_adam.jl"))

const DIAG_FT = Float32

"""Return CSC storage triples in exactly the same order as `SparseArrays.getnzval`."""
function storage_triples(A::AType) where {AType<:SparseMatrixCSC}
    rows = rowvals(A)
    vals = nonzeros(A)
    triples = Vector{Tuple{Int,Int,Int,eltype(vals)}}(undef, length(vals))
    ptr_out = 1
    @inbounds for col in axes(A, 2)
        for ptr in nzrange(A, col)
            triples[ptr_out] = (ptr, Int(rows[ptr]), Int(col), vals[ptr])
            ptr_out += 1
        end
    end
    return triples
end

"""Compute the bilinear Hamiltonian value from the raw sparse storage order."""
function bilinear_energy(A::AType, state::S) where {AType<:SparseMatrixCSC,S<:AbstractVector}
    total = zero(eltype(state))
    rows = rowvals(A)
    vals = nonzeros(A)
    @inbounds for col in axes(A, 2)
        scol = state[col]
        for ptr in nzrange(A, col)
            total += vals[ptr] * state[rows[ptr]] * scol
        end
    end
    return -DIAG_FT(0.5) * total
end

"""Assert that the graph sparse-gradient vector is aligned with adjacency storage."""
function validate_sparse_graph_gradient!(graph::G, free_state::S, nudged_state::S, buffers::B) where {G,S<:AbstractVector,B}
    A = sparse(II.adj(graph))
    triples = storage_triples(A)
    length(triples) == length(buffers.w) || error("gradient length does not match sparse storage")

    max_abs_error = 0.0
    reverse_mismatches = 0
    ptr_by_pair = Dict{Tuple{Int,Int},Int}()
    @inbounds for (ptr, row, col, _) in triples
        ptr_by_pair[(row, col)] = ptr
        expected = -DIAG_FT(0.5) * (nudged_state[row] * nudged_state[col] - free_state[row] * free_state[col])
        max_abs_error = max(max_abs_error, abs(Float64(buffers.w[ptr] - expected)))
    end

    @inbounds for (_, row, col, _) in triples
        rev = get(ptr_by_pair, (col, row), 0)
        rev == 0 && (reverse_mismatches += 1)
        rev != 0 && buffers.w[ptr_by_pair[(row, col)]] != buffers.w[rev] && (reverse_mismatches += 1)
    end

    eps = DIAG_FT(1f-3)
    finite_difference_error = 0.0
    for sample_idx in 1:min(20, length(triples))
        ptr, _, _, old = triples[sample_idx]
        nz = nonzeros(A)
        nz[ptr] = old + eps
        plus = bilinear_energy(A, nudged_state) - bilinear_energy(A, free_state)
        nz[ptr] = old - eps
        minus = bilinear_energy(A, nudged_state) - bilinear_energy(A, free_state)
        nz[ptr] = old
        fd = (plus - minus) / (DIAG_FT(2) * eps)
        finite_difference_error = max(finite_difference_error, abs(Float64(fd - buffers.w[ptr])))
    end

    return (;
        sparse_entries = length(triples),
        max_abs_error,
        reverse_mismatches,
        finite_difference_error,
    )
end

"""Assert that external image-projection gradients match `d(H+ - H-)/dW`."""
function validate_input_projection_gradient!(
    x::X,
    free_state::S,
    nudged_state::S,
    buffers::B,
) where {X<:AbstractVector,S<:AbstractVector,B}
    patches = LCL_PATCH_INPUT_IDXS[]
    mask = LCL_INPUT_MASK[]
    max_abs_error = 0.0
    max_mask_leak = 0.0
    @inbounds for hidden_idx in axes(buffers.w_input, 1)
        for input_idx in axes(buffers.w_input, 2)
            got = buffers.w_input[hidden_idx, input_idx]
            if mask[hidden_idx, input_idx]
                expected = -x[input_idx] * (nudged_state[hidden_idx] - free_state[hidden_idx])
                max_abs_error = max(max_abs_error, abs(Float64(got - expected)))
            else
                max_mask_leak = max(max_mask_leak, abs(Float64(got)))
            end
        end
    end

    eps = DIAG_FT(1f-3)
    finite_difference_error = 0.0
    checked = 0
    @inbounds for hidden_idx in eachindex(patches)
        for input_idx in patches[hidden_idx]
            checked += 1
            checked > 20 && break
            old = zero(DIAG_FT)
            hplus(w) = -w * x[input_idx] * nudged_state[hidden_idx]
            hminus(w) = -w * x[input_idx] * free_state[hidden_idx]
            fd = ((hplus(old + eps) - hminus(old + eps)) - (hplus(old - eps) - hminus(old - eps))) / (DIAG_FT(2) * eps)
            finite_difference_error = max(finite_difference_error, abs(Float64(fd - buffers.w_input[hidden_idx, input_idx])))
        end
        checked > 20 && break
    end

    return (;
        trainable_input_weights = count(mask),
        max_abs_error,
        max_mask_leak,
        finite_difference_error,
    )
end

"""Run LCL gradient and sparse-index sanity checks for a deterministic random state pair."""
function main()
    rng = Random.MersenneTwister(20260607)
    config = updated_config(
        InputFieldMNISTConfig();
        workers = 1,
        epochs = 0,
        batchsize = 1,
        train_per_class = 1,
        test_per_class = 1,
        train_eval_per_class = 0,
        hidden = lcl_hidden_side(LCL_WINDOW, LCL_STRIDE)^2,
        output_replicas = 1,
        β = DIAG_FT(0.3),
        outdir = joinpath(@__DIR__, "tmp_validate_lcl_gradient_and_indices"),
    )
    setup = build_layer(config)
    graph = setup.graph
    x = rand(rng, DIAG_FT, INPUT_DIM)
    free_state = DIAG_FT.(2 .* rand(rng, DIAG_FT, II.nstates(graph)) .- 1)
    nudged_state = DIAG_FT.(2 .* rand(rng, DIAG_FT, II.nstates(graph)) .- 1)
    buffers = input_field_gradient_buffer(graph, setup.input_hidden_w)

    accumulate_input_field_gradient!(graph, nudged_state, free_state, x, buffers, config.β)
    sparse_report = validate_sparse_graph_gradient!(graph, free_state, nudged_state, buffers)
    input_report = validate_input_projection_gradient!(x, free_state, nudged_state, buffers)
    bias_expected = .-(nudged_state .- free_state)
    bias_error = maximum(abs.(buffers.b .- bias_expected))

    println("sparse_report = ", sparse_report)
    println("input_report = ", input_report)
    println("bias_max_abs_error = ", bias_error)

    sparse_report.max_abs_error < 1f-6 || error("sparse graph gradient mismatch")
    sparse_report.reverse_mismatches == 0 || error("sparse reverse-pair mismatch")
    sparse_report.finite_difference_error < 5f-4 || error("sparse finite-difference mismatch")
    input_report.max_abs_error < 1f-6 || error("input projection gradient mismatch")
    input_report.max_mask_leak == 0.0 || error("input projection gradient leaked outside mask")
    input_report.finite_difference_error < 5f-4 || error("input projection finite-difference mismatch")
    bias_error < 1f-6 || error("bias gradient mismatch")
    println("LCL gradient/index validation passed")
end

main()
