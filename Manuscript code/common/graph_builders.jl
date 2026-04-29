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
    a1::Float64 = -2.0
    b1::Any = nothing
    c1::Float64 = 10.0
    landau_coeffs::Any = nothing
    landau_mode::Symbol = :coupled_diag
    landau_storage::Any = UniformArray
    outdir::String = raw"D:\Code\data\Manuscript\Demo1"
end

function update_params(p::ManuscriptParams; kwargs...)
    names = propertynames(p)
    values = Dict{Symbol,Any}(name => getproperty(p, name) for name in names)
    for (key, value) in kwargs
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
    b1 = get(coeffs, 4, nothing)
    E_barrier = abs(landau_energy(coeffs, 1))
    Epp_1 = landau_second_derivative(coeffs, 1)
    return (; fullsweep, anneal_time, pulse_time, relax_time, point_repeat,
        capture_interval1, capture_interval2, b1, E_barrier, Epp_1, landau_coeffs = coeffs)
end

manuscript_default_weight(; dc) = weightfunc_shell(1, 1, 1, 1, 0.1, 0.1; dc)
default_weight_generator() = InteractiveIsing.@WG manuscript_default_weight NN = 3

function landau_coefficients(p::ManuscriptParams)
    if !isnothing(p.landau_coeffs)
        coeffs = Dict(Int(order) => coeff for (order, coeff) in pairs(p.landau_coeffs))
        _validate_landau_orders(coeffs)
        return coeffs
    end

    b1 = isnothing(p.b1) ? -(p.a1 + 3 * p.c1) / 2 : p.b1
    coeffs = Dict(2 => p.a1, 4 => b1, 6 => p.c1)
    _validate_landau_orders(coeffs)
    return coeffs
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
        coeff isa Number || continue
        total += coeff * x^order
    end
    return total
end

function landau_second_derivative(coeffs, x)
    total = zero(float(x))
    for (order, coeff) in pairs(coeffs)
        coeff isa Number || continue
        order >= 2 || continue
        total += order * (order - 1) * coeff * x^(order - 2)
    end
    return total
end

function _poly_term(order::Integer; c = UniformArray(1), localpotential = g -> adj(g).diag)
    return PolynomialHamiltonian(order; c, localpotential)
end

function _normalized_landau_mode(mode::Symbol)
    mode == :coupled && return :coupled_diag
    mode == :independent && return :independent_lp
    return mode
end

function _coupled_diag_landau_hamiltonian(p::ManuscriptParams, coeffs)
    a2 = get(coeffs, 2, nothing)
    isnothing(a2) && error("Coupled diag Landau mode requires a 2nd-order coefficient, e.g. landau_coeffs = Dict(2=>a, 4=>b, 6=>c).")
    a2 isa Number || error("Coupled diag mode requires a scalar 2nd-order coefficient. Use landau_mode = :independent_lp for per-site Landau coefficients.")

    ham = Ising(b = StateLike(UniformArray, 0))
    ham += CoulombHamiltonian(scaling = p.Scale, screening = p.Screening, recalc = 1000)

    for order in sort(collect(keys(coeffs)))
        order == 2 && continue
        ham += _poly_term(order; c = coeffs[order] / a2)
    end

    return ham
end

function _independent_lp_landau_hamiltonian(p::ManuscriptParams, coeffs)
    localpotential = StateLike(p.landau_storage, 0)
    ham = Ising(b = StateLike(UniformArray, 0), localpotential = localpotential)
    ham += CoulombHamiltonian(scaling = p.Scale, screening = p.Screening, recalc = 1000)

    for order in sort(collect(keys(coeffs)))
        order == 2 && continue
        ham += _poly_term(order; localpotential)
    end

    return ham
end

function landau_hamiltonian(p::ManuscriptParams)
    coeffs = landau_coefficients(p)
    mode = _normalized_landau_mode(p.landau_mode)
    if mode == :coupled_diag
        return _coupled_diag_landau_hamiltonian(p, coeffs)
    elseif mode == :independent_lp
        return _independent_lp_landau_hamiltonian(p, coeffs)
    else
        error("Unknown landau_mode $(p.landau_mode). Use :coupled_diag or :independent_lp.")
    end
end

function _set_landau_value!(term, coeff)
    term.c[] = one(eltype(term.c))
    if coeff isa Number
        term.lp[] = coeff
    else
        term.lp .= coeff
    end
    return term
end

_poly_order(term::PolynomialHamiltonian) = typeof(term).parameters[1]

function apply_landau_coefficients!(g, p::ManuscriptParams)
    coeffs = landau_coefficients(p)
    mode = _normalized_landau_mode(p.landau_mode)

    if mode == :coupled_diag
        adj(g)[1, 1] = coeffs[2]
        return g
    end

    for term in InteractiveIsing.hamiltonians(g.hamiltonian)
        term isa PolynomialHamiltonian || continue
        coeff = get(coeffs, _poly_order(term), nothing)
        isnothing(coeff) && continue
        _set_landau_value!(term, coeff)
    end

    return g
end

function build_graph(p::ManuscriptParams; wg = default_weight_generator())
    g = IsingGraph(
        p.xL, p.yL, p.zL,
        Continuous(),
        wg,
        LatticeConstants(1.0, 1.0, 1.0),
        landau_hamiltonian(p),
        StateSet(-1.5f0, 1.5f0);
        periodic = (:x, :y),
        diag = StateLike(UniformArray),
    )
    normalize_adj_by_average_col!(g.adj, p.JIsing)
    apply_landau_coefficients!(g, p)
    temp!(g, p.Temp)
    return g
end
