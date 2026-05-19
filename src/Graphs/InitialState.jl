export local_potential_coefficients, local_potential_is_confining,
    local_potential_argmin

"""
    initState(g, initial_state)

Return an initialized state vector for `g`.

`initial_state === nothing` uses `initRandomState(g)`. A number fills the whole
state, an array is copied after a length check, and a function is called as
`initial_state(g)`. If the function mutates `g` and returns `nothing`, the
current graph state is copied.
"""
function initState(g::IsingGraph, initial_state)
    if isnothing(initial_state)
        return initRandomState(g)
    elseif initial_state isa Number
        return fill(convert(eltype(g), initial_state), size(graphstate(g)))
    elseif initial_state isa AbstractArray
        length(initial_state) == nstates(g) ||
            throw(ArgumentError("initial_state length $(length(initial_state)) does not match graph state length $(nstates(g))"))
        out = similar(graphstate(g))
        out[:] .= initial_state[:]
        return out
    elseif initial_state isa Function
        result = initial_state(g)
        isnothing(result) && return copy(graphstate(g))
        length(result) == nstates(g) ||
            throw(ArgumentError("initial_state(g) returned length $(length(result)); expected $(nstates(g))"))
        out = similar(graphstate(g))
        out[:] .= result[:]
        return out
    else
        throw(ArgumentError("Unsupported initial_state $(typeof(initial_state)); pass nothing, a number, an array, or a function."))
    end
end

@inline _finite_stateset(states) = isfinite(states[1]) && isfinite(states[end])

@inline function _init_layer_random_state!(dest, layer)
    dest .= rand(layer, length(dest))
    return dest
end

@inline function _init_layer_unbounded_state!(dest, g::IsingGraph, layer)
    for (local_pos, spin_idx) in enumerate(graphidxs(layer))
        @inbounds dest[local_pos] = _local_potential_initial_state(hamiltonian(g), g, spin_idx)
    end
    return dest
end

@inline function _init_layer_state!(dest, g::IsingGraph, layer)
    if statetype(layer) isa Continuous && !_finite_stateset(stateset(layer))
        return _init_layer_unbounded_state!(dest, g, layer)
    end
    return _init_layer_random_state!(dest, layer)
end

@inline function _local_potential_initial_state(ham, g::IsingGraph, spin_idx)
    coeffs = local_potential_coefficients(ham, g, spin_idx)
    local_potential_is_confining(coeffs) || return zero(eltype(g))
    return local_potential_argmin(coeffs)
end

"""
    local_potential_coefficients(hamiltonian, g, spin_idx)

Return coefficients of the spin-local polynomial energy at `spin_idx`.

The returned vector is in ascending power order, so `coeffs[k + 1]` multiplies
`x^k`. Non-local Hamiltonian terms contribute nothing by default.
"""
function local_potential_coefficients(ham, g::IsingGraph, spin_idx)
    T = eltype(g)
    coeffs = zeros(T, 9)
    local_potential_coefficients!(coeffs, ham, g, spin_idx)
    return coeffs
end

@inline local_potential_coefficients!(coeffs, ham, g::IsingGraph, spin_idx) = coeffs

@inline function local_potential_coefficients!(coeffs, hts::HamiltonianTerms, g::IsingGraph, spin_idx)
    for h in hamiltonians(hts)
        local_potential_coefficients!(coeffs, h, g, spin_idx)
    end
    return coeffs
end

@inline function local_potential_coefficients!(coeffs, h::PolynomialHamiltonian{Order}, g::IsingGraph, spin_idx) where {Order}
    length(coeffs) < Order + 1 && resize!(coeffs, Order + 1)
    coeffs[Order + 1] += convert(eltype(coeffs), h.c[]) * convert(eltype(coeffs), h.lp[spin_idx])
    return coeffs
end

@inline function local_potential_coefficients!(coeffs, h::MagField, g::IsingGraph, spin_idx)
    length(coeffs) < 2 && resize!(coeffs, 2)
    coeffs[2] += -convert(eltype(coeffs), h.c) * convert(eltype(coeffs), h.b[spin_idx])
    return coeffs
end

@inline function _trim_polynomial_degree(coeffs)
    degree = length(coeffs) - 1
    while degree > 0 && iszero(coeffs[degree + 1])
        degree -= 1
    end
    return degree
end

"""
    local_potential_is_confining(coeffs)

Return true when the polynomial with ascending coefficients goes to `+Inf` at
both `-Inf` and `+Inf`.
"""
function local_potential_is_confining(coeffs)
    degree = _trim_polynomial_degree(coeffs)
    degree > 0 || return false
    iseven(degree) || return false
    return coeffs[degree + 1] > zero(eltype(coeffs))
end

@inline function _eval_local_potential(coeffs, x)
    total = zero(promote_type(eltype(coeffs), typeof(x)))
    for c in Iterators.reverse(coeffs)
        total = total * x + c
    end
    return total
end

function _real_polynomial_roots(coeffs)
    degree = _trim_polynomial_degree(coeffs)
    T = eltype(coeffs)
    degree <= 0 && return T[]
    degree == 1 && return T[-coeffs[1] / coeffs[2]]

    companion = zeros(T, degree, degree)
    @inbounds for i in 2:degree
        companion[i, i - 1] = one(T)
    end
    leading = coeffs[degree + 1]
    @inbounds for i in 1:degree
        companion[i, degree] = -coeffs[i] / leading
    end

    roots = eigvals(companion)
    tol = sqrt(eps(T))
    real_roots = T[]
    for root in roots
        if abs(imag(root)) <= tol * max(one(T), abs(real(root)))
            push!(real_roots, convert(T, real(root)))
        end
    end
    return real_roots
end

"""
    local_potential_argmin(coeffs)

Return the global minimizer of a confining local polynomial energy represented
by ascending coefficients.
"""
function local_potential_argmin(coeffs)
    T = eltype(coeffs)
    local_potential_is_confining(coeffs) || return zero(T)

    degree = _trim_polynomial_degree(coeffs)
    derivative_coeffs = zeros(T, degree)
    @inbounds for power in 1:degree
        derivative_coeffs[power] = T(power) * coeffs[power + 1]
    end

    candidates = _real_polynomial_roots(derivative_coeffs)
    isempty(candidates) && return zero(T)

    best = first(candidates)
    best_energy = _eval_local_potential(coeffs, best)
    for candidate in candidates[2:end]
        energy = _eval_local_potential(coeffs, candidate)
        if energy < best_energy
            best = candidate
            best_energy = energy
        end
    end
    return best
end

"""
Initialize from a graph.

Bounded layers keep the historical random-from-`StateSet` behavior. Unbounded
continuous layers cannot be sampled uniformly, so they are initialized with a
finite local-potential minimum when the summed local potential is confining.
"""
function initRandomState(g)
    _state = similar(graphstate(g))
    # for layer in unshuffled(layers(g))
    for layer in layers(g)
        _init_layer_state!((@view _state[graphidxs(layer)]), g, layer)
    end
    return _state
end
