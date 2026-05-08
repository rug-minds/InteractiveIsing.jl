include(joinpath(@__DIR__, "parallel_sweep_example2.jl"))

###############################################################################
# Compare algorithms over time_fctr and step size
###############################################################################

comparison_base = MT.ManuscriptParams(;
    outdir = raw"D:\Code\data\Manuscript\Algorithm_comparison2",
    xL = 30,
    yL = 30,
    zL = 10,
    JIsing = 1.0,
    Scale = 2.0,
    Screening = 0.1,
    Temp = 0.15f0,
    Temp_aneal = 3f0,
    time_fctr = 1.0,
    Steps_1 = 6000,
    Amp1 = 3.0,
    nrepeats = 2,
    proposal_delta = 0.1,
    algorithm_name = :metropolis,
    algorithm_kwargs = (;),
    landau_mode = :independent,
    landau_coeffs = Dict(2 => -2.0, 4 => -12, 6 => 10, 8 => -1.0),
)

function _coeffvec_to_landau_dict(coeffvec; atol = 1e-6)
    out = Dict{Int,Float64}()
    for order in 0:length(coeffvec)-1
        coeff = Float64(coeffvec[order + 1])
        abs(coeff) <= atol && continue
        out[order] = coeff
    end
    return out
end

function _landau_dict_max_abs_diff(expected::Dict, actual::Dict)
    orders = union(keys(expected), keys(actual))
    isempty(orders) && return 0.0
    return maximum(abs(Float64(get(expected, order, 0.0)) - Float64(get(actual, order, 0.0))) for order in orders)
end

function landau_encoding_summary(g, p; spin_idx = 1, atol = 1e-6)
    expected = Dict(Int(order) => Float64(coeff) for (order, coeff) in pairs(MT.landau_coefficients(p)))
    actual = _coeffvec_to_landau_dict(
        InteractiveIsing.local_potential_coefficients(g.hamiltonian, g, spin_idx);
        atol,
    )
    maxdiff = _landau_dict_max_abs_diff(expected, actual)
    return (; expected, actual, maxdiff, ok = maxdiff <= atol)
end

function metropolis_paramsets(base; time_factors = (0.5, 1.0, 2.0, 5), deltas = (0.02,0.05, 0.1, 0.2, 0.5))
    return [
        MT.update_params(
            base;
            time_fctr = tf,
            algorithm_name = :metropolis,
            algorithm_kwargs = (;),
            proposal_delta = delta,
            outdir = joinpath(base.outdir, "metropolis", "tf=$(tf)_proposal=$(delta)"),
        )
        for tf in time_factors
        for delta in deltas
    ]
end

function local_langevin_paramsets(base; time_factors = (0.5, 1.0, 2.0, 5), stepsizes = (0.02f0, 0.05f0, 0.1f0, 0.2f0, 0.5f0))
    return [
        MT.update_params(
            base;
            time_fctr = tf,
            algorithm_name = :local_langevin,
            algorithm_kwargs = (; stepsize = step, adjusted = true),
            outdir = joinpath(base.outdir, "local_langevin", "tf=$(tf)_step=$(step)"),
        )
        for tf in time_factors
        for step in stepsizes
    ]
end

function global_langevin_paramsets(base; time_factors = (0.5, 1.0, 2.0), stepsizes = (0.005f0, 0.01f0, 0.02f0))
    return [
        MT.update_params(
            base;
            time_fctr = tf,
            algorithm_name = :global_langevin,
            algorithm_kwargs = (; stepsize = step, adjusted = false),
            outdir = joinpath(base.outdir, "global_langevin", "tf=$(tf)_step=$(step)"),
        )
        for tf in time_factors
        for step in stepsizes
    ]
end

function block_langevin_paramsets(base; time_factors = (0.5, 1.0, 2.0), stepsizes = (0.01f0, 0.02f0, 0.05f0), block_size = 128)
    return [
        MT.update_params(
            base;
            time_fctr = tf,
            algorithm_name = :block_langevin,
            algorithm_kwargs = (; stepsize = step, block_size = block_size, adjusted = false),
            outdir = joinpath(base.outdir, "block_langevin", "tf=$(tf)_step=$(step)"),
        )
        for tf in time_factors
        for step in stepsizes
    ]
end

function algorithm_comparison_paramsets(base)
    paramsets = MT.ManuscriptParams[]
    append!(paramsets, metropolis_paramsets(base))
    append!(paramsets, local_langevin_paramsets(base))
    # append!(paramsets, global_langevin_paramsets(base))
    # append!(paramsets, block_langevin_paramsets(base))
    return paramsets
end

function print_algorithm_summary(results)
    for item in results
        sanity = landau_encoding_summary(item.graph, item.params)
        println()
        println("Saved: ", item.paths.xlsx_path)
        println("Algorithm: ", item.params.algorithm_name)
        println("time_fctr: ", item.params.time_fctr)
        println("proposal_delta: ", item.params.proposal_delta)
        println("algorithm_kwargs: ", item.params.algorithm_kwargs)
        println("Landau encoding ok: ", sanity.ok, " (max diff = ", sanity.maxdiff, ")")
        !sanity.ok && println("expected vs actual Landau coefficients: ", sort(collect(sanity.expected)), " vs ", sort(collect(sanity.actual)))
    end
    return results
end

function run_algorithm_comparison(; max_inflight = 4, capture = false)
    paramsets = algorithm_comparison_paramsets(comparison_base)
    println("Prepared $(length(paramsets)) algorithm-comparison runs.")
    results = run_packaged_pulse_sweep_batched(paramsets; max_inflight, capture)
    print_algorithm_summary(results)
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_algorithm_comparison(; max_inflight = 4, capture = false)
end
