Base.@kwdef struct ManuscriptParams
    xL::Int = 40
    yL::Int = 40
    zL::Int = 10
    JIsing::Float64 = 1.0
    Scale::Float64 = 1.0
    Screening::Float64 = 0.01
    Temp::Float32 = 0.5f0
    Temp_aneal::Float32 = 5.0f0
    time_fctr::Float64 = 1.0
    Steps_1::Int = 6000
    Amp1::Float64 = 10.0
    nrepeats::Int = 2
    linear_field_coeff::Float64 = 1.0
    defect_field_coeff::Float64 = 0.0
    proposal_delta::Any = nothing
    algorithm_name::Symbol = :default
    algorithm_kwargs::Any = (;)
    a1::Float64 = -2.0
    b1::Any = nothing
    c1::Float64 = 10.0
    d1::Float64 = 0.0
    e1::Float64 = 0.0
    landau_coeffs::Any = nothing
    landau_mode::Symbol = :independent
    apply_weak_landau_disorder::Bool = false
    coeff2_disorder_scale::Float64 = 0.5
    coeff4_disorder_scale::Float64 = 1.0
    coeff6_disorder_scale::Float64 = 1.0
    coeff8_disorder_scale::Float64 = 0.2
    coeff10_disorder_scale::Float64 = 0.2
    disorder_seed::Any = 0
    coulomb_recalc::Int = 1000
    log_diagnostics::Bool = true
    capture::Bool = false
    save_figures::Bool = true
    save_xlsx::Bool = true
    state_min::Float32 = -1.5f0
    state_max::Float32 = 1.5f0
    outdir::String = raw"D:\Code\data\Manuscript\Demo1"
end

function update_params(p::ManuscriptParams; kwargs...)
    names = propertynames(p)
    values = Dict{Symbol,Any}(name => getproperty(p, name) for name in names)
    for (key, value) in kwargs
        key in names || throw(ArgumentError("Unknown ManuscriptParams field $(repr(key))."))
        values[key] = value
    end
    nt = NamedTuple{names}(Tuple(values[name] for name in names))
    return ManuscriptParams(; nt...)
end

function derived_params(p::ManuscriptParams)
    coeffs = landau_coefficients(p)
    fullsweep = p.xL * p.yL * p.zL
    anneal_time = p.time_fctr * fullsweep * p.Steps_1
    pulse_time = p.time_fctr * fullsweep * p.Steps_1
    relax_time = p.time_fctr / 2 * fullsweep * p.Steps_1
    point_repeat = fullsweep * p.time_fctr
    capture_interval1 = pulse_time / (p.nrepeats * 4)
    capture_interval2 = relax_time / 2
    representative = _representative_landau_coefficients(coeffs)
    b1 = get(representative, 4, nothing)
    barrier = estimate_landau_barrier(representative; Pmin = p.state_min, Pmax = p.state_max)
    E_barrier = barrier.ΔF
    Epp_1 = landau_second_derivative(representative, barrier.P0)
    return (; fullsweep, anneal_time, pulse_time, relax_time, point_repeat,
        capture_interval1, capture_interval2, b1, E_barrier, Epp_1,
        landau_barrier = barrier, landau_coeffs = coeffs)
end

manuscript_default_weight(; dc) = weightfunc_shell(1, 1, 1, 1, 0.1, 0.1; dc)
default_weight_generator() = InteractiveIsing.@WG manuscript_default_weight NN = 3

function landau_coefficients(p::ManuscriptParams)
    if !isnothing(p.landau_coeffs)
        coeffs = Dict(Int(order) => coeff for (order, coeff) in pairs(p.landau_coeffs))
        _validate_landau_orders(coeffs)
        return _apply_landau_disorder(coeffs, p)
    end

    b1 = isnothing(p.b1) ? -(p.a1 + 3 * p.c1) / 2 : p.b1
    coeffs = Dict(2 => p.a1, 4 => b1, 6 => p.c1, 8 => p.d1, 10 => p.e1)
    _validate_landau_orders(coeffs)
    return _apply_landau_disorder(coeffs, p)
end

_coefficient_mean(coeff) = coeff isa Number ? Float64(coeff) : mean(Float64.(coeff))

function _representative_landau_coefficients(coeffs)
    return Dict(Int(order) => _coefficient_mean(coeff) for (order, coeff) in pairs(coeffs))
end

function _apply_landau_disorder(coeffs, p::ManuscriptParams)
    p.apply_weak_landau_disorder || return coeffs
    rng = isnothing(p.disorder_seed) ? Random.default_rng() : MersenneTwister(p.disorder_seed)
    nspins = p.xL * p.yL * p.zL
    scales = Dict(
        2 => p.coeff2_disorder_scale,
        4 => p.coeff4_disorder_scale,
        6 => p.coeff6_disorder_scale,
        8 => p.coeff8_disorder_scale,
        10 => p.coeff10_disorder_scale,
    )
    return Dict(
        order => fill(Float32(_coefficient_mean(coeff)), nspins) .+
            Float32(get(scales, order, 0.0)) .* randn(rng, Float32, nspins)
        for (order, coeff) in pairs(coeffs)
    )
end

function _validate_landau_orders(coeffs)
    for order in keys(coeffs)
        iseven(order) || error("Landau polynomial order must be even; got order $order.")
        order >= 2 || error("Landau polynomial order must be at least 2; got order $order.")
    end
    return coeffs
end

function landau_energy(coeffs, x)
    total = zero(float(x))
    for (order, coeff) in pairs(coeffs)
        total += _coefficient_mean(coeff) * x^order
    end
    return total
end

function landau_second_derivative(coeffs, x)
    total = zero(float(x))
    for (order, coeff) in pairs(coeffs)
        order >= 2 || continue
        total += order * (order - 1) * _coefficient_mean(coeff) * x^(order - 2)
    end
    return total
end

function estimate_landau_barrier(coeffs; Pmin = -1.5, Pmax = 1.5, ngrid = 20001)
    xs = range(Float64(Pmin), Float64(Pmax), length = ngrid)
    ys = [landau_energy(coeffs, x) for x in xs]
    minima = [i for i in 2:(length(xs) - 1) if ys[i] <= ys[i - 1] && ys[i] <= ys[i + 1]]
    maxima = [i for i in 2:(length(xs) - 1) if ys[i] >= ys[i - 1] && ys[i] >= ys[i + 1]]
    positive_minima = filter(i -> xs[i] > 0, minima)
    isempty(positive_minima) && error("No positive local minimum found in the Landau scan.")
    well_idx = positive_minima[argmin(ys[positive_minima])]
    between = filter(i -> 0 <= xs[i] <= xs[well_idx], maxima)
    barrier_idx = isempty(between) ? argmin(abs.(xs)) : between[argmax(ys[between])]
    return (; P0 = xs[well_idx], Ps = xs[barrier_idx],
        ΔF = ys[barrier_idx] - ys[well_idx],
        Ewell = ys[well_idx], Ebarrier = ys[barrier_idx])
end

_landau_localpotential(coeff::Number) = UniformArray(Float32(coeff))
_landau_localpotential(coeff) = Float32.(vec(coeff))

function _independent_landau_hamiltonian(p::ManuscriptParams, coeffs)
    quadratic = get(coeffs, 2, 0.0)
    ham = Ising(
        b = UniformArray(Float32(p.linear_field_coeff)),
        localpotential = _landau_localpotential(quadratic),
    )
    ham += CoulombHamiltonian(
        scaling = p.Scale,
        screening = p.Screening,
        recalc = p.coulomb_recalc,
    )
    for order in sort(collect(keys(coeffs)))
        order == 2 && continue
        ham += PolynomialHamiltonian(
            order;
            c = UniformArray(1.0f0),
            localpotential = _landau_localpotential(coeffs[order]),
        )
    end
    return ham
end

function landau_hamiltonian(p::ManuscriptParams)
    coeffs = landau_coefficients(p)
    p.landau_mode in (:independent, :independent_lp) ||
        error("Unknown landau_mode $(p.landau_mode). Only :independent is supported.")
    return _independent_landau_hamiltonian(p, coeffs)
end

function _set_landau_lp!(term, coeff)
    if coeff isa Number
        term.lp[] = coeff
    else
        length(coeff) == length(term.lp) ||
            throw(DimensionMismatch("Landau coefficient field length $(length(coeff)) does not match localpotential length $(length(term.lp))."))
        term.lp .= reshape(vec(coeff), size(term.lp))
    end
    return term
end

function _set_landau_value!(term, coeff)
    term.c[] = one(eltype(term.c))
    return _set_landau_lp!(term, coeff)
end

_poly_order(term::PolynomialHamiltonian) = typeof(term).parameters[1]

function apply_landau_coefficients!(g, p::ManuscriptParams)
    coeffs = landau_coefficients(p)
    p.landau_mode in (:independent, :independent_lp) ||
        error("Unknown landau_mode $(p.landau_mode). Only :independent is supported.")

    for term in InteractiveIsing.hamiltonians(g.hamiltonian)
        term isa PolynomialHamiltonian || continue
        coeff = get(coeffs, _poly_order(term), nothing)
        isnothing(coeff) && continue
        _set_landau_value!(term, coeff)
    end

    return g
end

function build_graph(p::ManuscriptParams; wg = default_weight_generator())
    proposer_args = isnothing(p.proposal_delta) ? () : (LocalProposer(p.proposal_delta),)
    g = IsingGraph(
        p.xL, p.yL, p.zL,
        Continuous(),
        proposer_args...,
        wg,
        LatticeConstants(1.0, 1.0, 1.0),
        landau_hamiltonian(p),
        StateSet(p.state_min, p.state_max);
        periodic = (:x, :y),
        diag = StateLike(UniformArray, 0.0f0),
    )
    normalize_adj_by_average_col!(g.adj, p.JIsing)
    temp!(g, p.Temp)
    return g
end

function reduced_parameter_summary(g, p::ManuscriptParams)
    d = derived_params(p)
    barrier = d.landau_barrier
    A = hasproperty(adj(g), :sp) ? adj(g).sp : adj(g)
    colsums = [sum(abs, @view A[:, j]) for j in axes(A, 2)]
    SJ = mean(colsums)
    P0 = barrier.P0
    ΔF = barrier.ΔF
    rows = NamedTuple[]
    add!(section, key, value; note = "") = push!(rows, (; section, key, value, note))
    add!("input", "JIsing", p.JIsing)
    add!("input", "Scale", p.Scale)
    add!("input", "Screening", p.Screening)
    add!("input", "linear_field_coeff", p.linear_field_coeff)
    add!("input", "defect_field_coeff", p.defect_field_coeff)
    add!("direct", "P0", P0)
    add!("direct", "Ps", barrier.Ps)
    add!("direct", "DeltaF_barrier", ΔF)
    add!("direct", "S_J", SJ)
    add!("direct", "S_J_min", minimum(colsums))
    add!("direct", "S_J_max", maximum(colsums))
    add!("direct", "Lambda_int", (P0^2 * SJ) / ΔF)
    add!("direct", "Lambda_barrier", ΔF / (P0^2 * SJ))
    add!("direct", "Lambda_field", abs(p.linear_field_coeff) / (P0 * SJ))
    add!("direct", "Lambda_defect", abs(p.defect_field_coeff) / (P0 * SJ))
    add!("direct", "Theta_field", abs(P0 * p.linear_field_coeff) / ΔF)
    add!("direct", "Theta_defect", abs(P0 * p.defect_field_coeff) / ΔF)
    return DataFrame(rows)
end

function select_dynamics(g, p::ManuscriptParams)
    name = p.algorithm_name
    kwargs = p.algorithm_kwargs

    name in (:default, :metropolis) && return g.default_algorithm
    name == :local_langevin && return LocalLangevin(; kwargs...)
    name == :global_langevin && return GlobalLangevin(; kwargs...)
    name == :block_langevin && return BlockLangevin(; kwargs...)

    error("Unknown algorithm_name $(repr(name)). Use :default, :metropolis, :local_langevin, :global_langevin, or :block_langevin.")
end
