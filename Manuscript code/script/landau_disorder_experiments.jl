include(joinpath(@__DIR__, "parallel_sweep_example2.jl"))

using Random

###############################################################################
# Spatially disordered Landau experiments
#
# Use `landau_mode = :independent` with state-length coefficient vectors to
# assign a different local polynomial to different dipoles. This is the right mode for:
#   1. weak quenched disorder in a/b/c/d/e
#   2. pinned dipoles / hard regions
#   3. domains, layers, or patterned regions with different Landau wells
###############################################################################

function coeffs_246(a, c)
    b = -(a + 3c) / 2
    return Dict(2 => a, 4 => b, 6 => c)
end

disorder_base = MT.ManuscriptParams(;
    outdir = raw"D:\Code\data\Manuscript\Landau_disorder",
    xL = 40,
    yL = 40,
    zL = 10,
    JIsing = 1.0,
    Scale = 1.0,
    Screening = 0.05,
    Temp = 0.35f0,
    Temp_aneal = 3f0,
    time_fctr = 1.0,
    Steps_1 = 6000,
    Amp1 = 3.0,
    nrepeats = 2,
    proposal_delta = nothing,
    algorithm_name = :local_langevin,
    algorithm_kwargs = (; stepsize = 0.05f0, adjusted = true),
    landau_mode = :independent,
    landau_storage = Vector,
)

gridshape(p) = (p.xL, p.yL, p.zL)
statelen(p) = p.xL * p.yL * p.zL

function coefficient_field(p, value)
    return fill(Float32(value), statelen(p))
end

function coefficient_fields(p, coeffs::Dict)
    return Dict(Int(order) => coefficient_field(p, coeff) for (order, coeff) in pairs(coeffs))
end

function coefficient_stats(coeffs::Dict)
    parts = String[]
    for order in sort(collect(keys(coeffs)))
        coeff = coeffs[order]
        if coeff isa Number
            push!(parts, "P^$(order): $(Float64(coeff))")
        else
            push!(parts, "P^$(order): [$(Float64(minimum(coeff))), $(Float64(maximum(coeff)))]")
        end
    end
    return join(parts, ", ")
end

function box_mask(p; xrange = 1:p.xL, yrange = 1:p.yL, zrange = 1:p.zL)
    mask3 = falses(gridshape(p)...)
    mask3[xrange, yrange, zrange] .= true
    return vec(mask3)
end

function layer_mask(p, zlayer::Integer)
    1 <= zlayer <= p.zL || error("zlayer must lie in 1:$(p.zL), got $(zlayer).")
    return box_mask(p; zrange = zlayer:zlayer)
end

function cylinder_mask(p; center = ((p.xL + 1) / 2, (p.yL + 1) / 2), radius = min(p.xL, p.yL) / 6, zrange = 1:p.zL)
    cx, cy = center
    r2 = radius^2
    mask3 = falses(gridshape(p)...)
    for z in zrange, y in 1:p.yL, x in 1:p.xL
        if (x - cx)^2 + (y - cy)^2 <= r2
            mask3[x, y, z] = true
        end
    end
    return vec(mask3)
end

function with_quenched_noise(base_fields::Dict; rng = Random.default_rng(), sigma = Dict(2 => 0.1, 4 => 0.2, 6 => 0.2, 8 => 0.0, 10 => 0.0))
    out = Dict{Int,Any}()
    for (order, field) in pairs(base_fields)
        if field isa Number
            base = Float32(field)
            out[order] = base + Float32(get(sigma, order, 0.0)) * randn(rng, Float32)
        else
            noisy = Float32.(field)
            s = Float32(get(sigma, order, 0.0))
            if !iszero(s)
                noisy .+= s .* randn(rng, Float32, size(noisy))
            end
            out[order] = noisy
        end
    end
    return out
end

function with_region_override(base_fields::Dict, mask, override_coeffs::Dict)
    out = Dict{Int,Any}(order => copy(field) for (order, field) in pairs(base_fields))
    for (order, coeff) in pairs(override_coeffs)
        haskey(out, order) || error("Override order $order not present in base fields.")
        out[order][mask] .= Float32(coeff)
    end
    return out
end

function with_piecewise_domain(base_fields::Dict, masks_and_coeffs)
    out = Dict{Int,Any}(order => copy(field) for (order, field) in pairs(base_fields))
    for (mask, coeffs) in masks_and_coeffs
        for (order, coeff) in pairs(coeffs)
            haskey(out, order) || error("Domain order $order not present in base fields.")
            out[order][mask] .= Float32(coeff)
        end
    end
    return out
end

function disorder_paramsets(base)
    uniform = coefficient_fields(base, coeffs_246(-2.0, 10.0))
    configs = [
        (
            name = "weak_random_abc",
            coeffs = with_quenched_noise(
                uniform;
                rng = MersenneTwister(11),
                sigma = Dict(2 => 0.10, 4 => 0.25, 6 => 0.15),
            ),
        ),
        (
            name = "stronger_random_abc",
            coeffs = with_quenched_noise(
                uniform;
                rng = MersenneTwister(12),
                sigma = Dict(2 => 0.20, 4 => 0.50, 6 => 0.30),
            ),
        ),
    ]

    return [
        MT.update_params(
            base;
            outdir = joinpath(base.outdir, "quenched_disorder", item.name),
            landau_coeffs = item.coeffs,
        )
        for item in configs
    ]
end

function pinned_paramsets(base)
    uniform = coefficient_fields(base, coeffs_246(-2.0, 10.0))
    central_pin = cylinder_mask(base; radius = min(base.xL, base.yL) / 7, zrange = 1:base.zL)
    top_patch = box_mask(base; xrange = 10:18, yrange = 10:18, zrange = base.zL:base.zL)

    configs = [
        (
            name = "central_hard_barrier",
            coeffs = with_region_override(
                uniform,
                central_pin,
                Dict(2 => 2.5, 4 => 18.0, 6 => 14.0),
            ),
        ),
        (
            name = "top_patch_deep_well",
            coeffs = with_region_override(
                uniform,
                top_patch,
                Dict(2 => -4.0, 4 => -18.0, 6 => 14.0),
            ),
        ),
    ]

    return [
        MT.update_params(
            base;
            outdir = joinpath(base.outdir, "pinned_regions", item.name),
            landau_coeffs = item.coeffs,
        )
        for item in configs
    ]
end

function domain_paramsets(base)
    uniform = coefficient_fields(base, coeffs_246(-2.0, 10.0))
    midx = max(1, div(base.xL, 2))
    midz = max(1, div(base.zL, 2))
    left_half = box_mask(base; xrange = 1:midx)
    right_half = box_mask(base; xrange = (midx + 1):base.xL)
    middle_layer = layer_mask(base, midz)
    left_coeffs = Dict(2 => -2.0, 4 => -14.0, 6 => 10.0)
    right_coeffs = Dict(2 => -1.0, 4 => -10.0, 6 => 8.0)
    wall_coeffs = Dict(2 => 1.5, 4 => 10.0, 6 => 8.0)

    configs = [
        (
            name = "left_right_two_domain",
            coeffs = with_piecewise_domain(
                uniform,
                [
                    (left_half, left_coeffs),
                    (right_half, right_coeffs),
                ],
            ),
        ),
        (
            name = "middle_layer_harder",
            coeffs = with_piecewise_domain(
                uniform,
                [
                    (left_half, left_coeffs),
                    (right_half, right_coeffs),
                    (middle_layer, wall_coeffs),
                ],
            ),
        ),
    ]

    return [
        MT.update_params(
            base;
            outdir = joinpath(base.outdir, "domain_patterns", item.name),
            landau_coeffs = item.coeffs,
        )
        for item in configs
    ]
end

function print_landau_disorder_summary(results)
    for item in results
        println()
        println("Saved: ", item.paths.xlsx_path)
        println("Run folder: ", item.params.outdir)
        println("Algorithm: ", item.params.algorithm_name, " ", item.params.algorithm_kwargs)
        println("Landau field ranges: ", coefficient_stats(item.params.landau_coeffs))
    end
    return results
end

function run_landau_disorder_suite(paramsets; max_inflight = 2, capture = false)
    println("Prepared $(length(paramsets)) spatial-Landau runs.")
    results = run_packaged_pulse_sweep_batched(paramsets; max_inflight, capture)
    print_landau_disorder_summary(results)
    return results
end

function run_quenched_disorder(; max_inflight = 2, capture = false)
    return run_landau_disorder_suite(disorder_paramsets(disorder_base); max_inflight, capture)
end

function run_pinned_regions(; max_inflight = 2, capture = false)
    return run_landau_disorder_suite(pinned_paramsets(disorder_base); max_inflight, capture)
end

function run_domain_patterns(; max_inflight = 2, capture = false)
    return run_landau_disorder_suite(domain_paramsets(disorder_base); max_inflight, capture)
end

if abspath(PROGRAM_FILE) == @__FILE__
    # Pick one family at a time.
    run_quenched_disorder(; max_inflight = 2, capture = false)
    # run_pinned_regions(; max_inflight = 2, capture = false)
    # run_domain_patterns(; max_inflight = 2, capture = false)
end
