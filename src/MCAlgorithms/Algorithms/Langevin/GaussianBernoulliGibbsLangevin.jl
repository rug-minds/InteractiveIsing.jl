export GaussianBernoulliGibbsLangevin

"""
    GaussianBernoulliGibbsLangevin(; stepsize = 0.01, langevin_steps = 5,
        group_steps = 1, adjusted = false, adjust_step = 0, burnin = 0)

Specialized Gibbs-Langevin sampler for the `GaussianBernoulli` RBM Hamiltonian.

Each outer step first runs `langevin_steps` visible Langevin proposals
conditioned on the previous hidden state, then samples the hidden layer from
`p(h | v)`. When `adjusted=true`, the full visible-hidden proposal is accepted
or rejected with the paper's joint Metropolis-Hastings correction after
`adjust_step` accepted-without-adjustment warmup steps.
"""
struct GaussianBernoulliGibbsLangevin{T<:Real} <: IsingMCAlgorithm
    stepsize::T
    langevin_steps::Int
    group_steps::Int
    adjusted::Bool
    adjust_step::Int
    burnin::Int
end

function GaussianBernoulliGibbsLangevin(;
    stepsize = 0.01,
    langevin_steps::Integer = 5,
    group_steps::Integer = 1,
    adjusted::Bool = false,
    adjust_step::Integer = 0,
    burnin::Integer = 0,
)
    return GaussianBernoulliGibbsLangevin(
        stepsize,
        max(1, Int(langevin_steps)),
        max(1, Int(group_steps)),
        adjusted,
        max(0, Int(adjust_step)),
        max(0, Int(burnin)),
    )
end

function _gb_find_hterm(hamiltonian)
    if isdefined(@__MODULE__, :GaussianBernoulli) && hamiltonian isa GaussianBernoulli
        return hamiltonian
    end

    if isdefined(@__MODULE__, :hamiltonians) && applicable(hamiltonians, hamiltonian)
        for hterm in hamiltonians(hamiltonian)
            if hterm isa GaussianBernoulli
                return hterm
            end
        end
    end

    throw(ArgumentError("GaussianBernoulliGibbsLangevin requires a GaussianBernoulli Hamiltonian term."))
end

@inline _gb_cosine_stepsize(k::Integer, K::Integer, α0) =
    α0 * (one(α0) + cos(convert(typeof(α0), k) * convert(typeof(α0), pi) / convert(typeof(α0), K))) / convert(typeof(α0), 2)

function _gb_fill_alphas!(alphas::AbstractVector, K::Integer, α0)
    if length(alphas) < K
        resize!(alphas, K)
    end
    @inbounds for k in 1:K
        alphas[k] = _gb_cosine_stepsize(k, K, α0)
    end
    return @view alphas[1:K]
end

function _gb_visible_conditional_mean!(dest, hterm, h::AbstractVector)
    n_visible, n_hidden = size(hterm.w)
    @inbounds for i in 1:n_visible
        wh = zero(eltype(dest))
        for j in 1:n_hidden
            wh += hterm.w[i, j] * h[j]
        end
        dest[i] = hterm.μ[i] + wh
    end
    return dest
end

function _gb_conditional_langevin!(
    rng::AbstractRNG,
    candidate_v::AbstractVector,
    grad::AbstractVector,
    hterm,
    h::AbstractVector,
    alphas::AbstractVector,
)
    T = eltype(candidate_v)
    @inbounds for α in alphas
        visible_energy_gradient!(grad, hterm, candidate_v, h)
        noise_scale = sqrt(max(zero(T), T(2) * T(α)))
        for i in eachindex(candidate_v)
            candidate_v[i] = candidate_v[i] - T(α) * grad[i] + noise_scale * randn(rng, T)
        end
    end
    return candidate_v
end

function _gb_langevin_proposal_stats!(
    mean::AbstractVector,
    variance::AbstractVector,
    hterm,
    start_v::AbstractVector,
    h::AbstractVector,
    alphas::AbstractVector,
    conditional_mean::AbstractVector,
)
    _gb_visible_conditional_mean!(conditional_mean, hterm, h)
    @inbounds for i in eachindex(start_v)
        σ2i = exp(hterm.logσ2[i])
        β_next = one(eltype(mean))
        a_num = zero(eltype(mean))
        var_i = zero(eltype(mean))
        for k in Iterators.reverse(eachindex(alphas))
            βk = β_next
            αk = alphas[k]
            a_num += βk * αk
            var_i += 2 * αk * βk * βk
            β_next *= one(β_next) - αk / σ2i
        end
        mean[i] = β_next * start_v[i] + (a_num / σ2i) * conditional_mean[i]
        variance[i] = var_i
    end
    return mean, variance
end

function _gb_log_gaussian_diag(x::AbstractVector, mean::AbstractVector, variance::AbstractVector)
    total = zero(promote_type(eltype(x), eltype(mean), eltype(variance)))
    @inbounds for i in eachindex(x)
        variance[i] > zero(variance[i]) || return oftype(total, -Inf)
        dx = x[i] - mean[i]
        total -= dx * dx / (2 * variance[i])
    end
    return total
end

function _gb_log_hidden_prob(hterm, v::AbstractVector, h::AbstractVector, logits::AbstractVector)
    hidden_logits!(logits, hterm, v)
    total = zero(eltype(logits))
    @inbounds for j in eachindex(h)
        total += h[j] > eltype(h)(0.5) ? -_gb_softplus(-logits[j]) : -_gb_softplus(logits[j])
    end
    return total
end

function _gb_log_acceptance!(
    hterm,
    old_v::AbstractVector,
    old_h::AbstractVector,
    new_v::AbstractVector,
    new_h::AbstractVector,
    alphas::AbstractVector,
    forward_mean::AbstractVector,
    reverse_mean::AbstractVector,
    proposal_variance::AbstractVector,
    conditional_mean::AbstractVector,
    logits::AbstractVector,
)
    _gb_langevin_proposal_stats!(forward_mean, proposal_variance, hterm, old_v, old_h, alphas, conditional_mean)
    log_forward_v = _gb_log_gaussian_diag(new_v, forward_mean, proposal_variance)
    log_forward_h = _gb_log_hidden_prob(hterm, new_v, new_h, logits)

    _gb_langevin_proposal_stats!(reverse_mean, proposal_variance, hterm, new_v, new_h, alphas, conditional_mean)
    log_reverse_v = _gb_log_gaussian_diag(old_v, reverse_mean, proposal_variance)
    log_reverse_h = _gb_log_hidden_prob(hterm, old_v, old_h, logits)

    return -joint_energy(hterm, new_v, new_h) + joint_energy(hterm, old_v, old_h) +
           log_reverse_v + log_reverse_h - log_forward_v - log_forward_h
end

function Processes.init(algorithm::GaussianBernoulliGibbsLangevin, context)
    (; model) = context
    hamiltonian = init!(model.hamiltonian, model)
    hterm = _gb_find_hterm(hamiltonian)
    _gb_validate_layers(hterm, model)

    rng = Random.MersenneTwister()
    SType = eltype(model)
    visible = _gb_visible_view(hterm, model)
    hidden = _gb_hidden_view(hterm, model)
    n_visible = length(visible)
    n_hidden = length(hidden)
    K = max(1, algorithm.langevin_steps)

    stepsize_default = SType(algorithm.stepsize)
    stepsize = Ref(SType(_langevin_unwrap_ref(_langevin_context_value(context, :stepsize, stepsize_default))))

    old_v = similar(visible, SType, n_visible)
    old_h = similar(hidden, SType, n_hidden)
    new_v = similar(visible, SType, n_visible)
    new_h = similar(hidden, SType, n_hidden)
    grad = similar(visible, SType, n_visible)
    alphas = Vector{SType}(undef, K)
    forward_mean = similar(visible, SType, n_visible)
    reverse_mean = similar(visible, SType, n_visible)
    proposal_variance = similar(visible, SType, n_visible)
    conditional_mean = similar(visible, SType, n_visible)
    logits = similar(hidden, SType, n_hidden)
    at_idxs = Vector{Int}(undef, n_visible + n_hidden)
    layer_idxs = Vector{Int}(undef, n_visible + n_hidden)
    from_vals = Vector{SType}(undef, n_visible + n_hidden)
    to_vals = Vector{SType}(undef, n_visible + n_hidden)

    v_idxs = visible_indices(hterm, model)
    h_idxs = hidden_indices(hterm, model)
    @inbounds for (pos, idx) in enumerate(v_idxs)
        at_idxs[pos] = idx
        layer_idxs[pos] = hterm.visible_layer
    end
    @inbounds for (local_pos, idx) in enumerate(h_idxs)
        pos = n_visible + local_pos
        at_idxs[pos] = idx
        layer_idxs[pos] = hterm.hidden_layer
    end

    proposal = MultiSpinProposal(Int[], SType[], SType[], Int[], false)
    accepted = 0
    attempted = 0
    acceptance_rate = zero(SType)
    step_idx = Ref(0)

    return (;
        model,
        hamiltonian,
        hterm,
        rng,
        stepsize,
        group_steps = Ref(algorithm.group_steps),
        adjusted = Ref(algorithm.adjusted),
        adjust_step = Ref(algorithm.adjust_step),
        burnin = Ref(algorithm.burnin),
        langevin_steps = Ref(K),
        old_v,
        old_h,
        new_v,
        new_h,
        grad,
        alphas,
        forward_mean,
        reverse_mean,
        proposal_variance,
        conditional_mean,
        logits,
        at_idxs,
        layer_idxs,
        from_vals,
        to_vals,
        proposal,
        accepted,
        attempted,
        acceptance_rate,
        step_idx,
    )
end

function Processes.step!(algorithm::GaussianBernoulliGibbsLangevin, context)
    (; model, hamiltonian, hterm, rng, stepsize, group_steps, adjusted,
        adjust_step, burnin, langevin_steps, old_v, old_h, new_v, new_h, grad,
        alphas, forward_mean, reverse_mean, proposal_variance, conditional_mean,
        logits, at_idxs, layer_idxs, from_vals, to_vals, step_idx) = context

    visible = _gb_visible_view(hterm, model)
    hidden = _gb_hidden_view(hterm, model)
    SType = eltype(model)
    n_visible = length(visible)
    n_hidden = length(hidden)
    n_total = n_visible + n_hidden
    n_group_steps = max(1, group_steps[])
    K = max(1, langevin_steps[])
    α0 = max(SType(stepsize[]), eps(SType))
    αs = _gb_fill_alphas!(alphas, K, α0)

    accepted = 0
    attempted = 0
    log_acceptance = zero(SType)
    proposal = MultiSpinProposal(Int[], SType[], SType[], Int[], false)

    for _ in 1:n_group_steps
        attempted += 1
        step_idx[] += 1

        copyto!(old_v, visible)
        copyto!(old_h, hidden)
        copyto!(new_v, old_v)

        _gb_conditional_langevin!(rng, new_v, grad, hterm, old_h, αs)
        sample_hidden!(rng, new_h, hterm, new_v)

        use_adjustment = adjusted[] && step_idx[] > adjust_step[]
        accept_move = true
        if use_adjustment
            log_acceptance = _gb_log_acceptance!(
                hterm,
                old_v,
                old_h,
                new_v,
                new_h,
                αs,
                forward_mean,
                reverse_mean,
                proposal_variance,
                conditional_mean,
                logits,
            )
            accept_move = isfinite(log_acceptance) &&
                          (log_acceptance >= zero(SType) || log(rand(rng, SType)) < log_acceptance)
        end

        @inbounds for i in 1:n_visible
            from_vals[i] = old_v[i]
            to_vals[i] = new_v[i]
        end
        @inbounds for j in 1:n_hidden
            pos = n_visible + j
            from_vals[pos] = old_h[j]
            to_vals[pos] = new_h[j]
        end

        idxs_view = @view at_idxs[1:n_total]
        from_view = @view from_vals[1:n_total]
        to_view = @view to_vals[1:n_total]
        layer_view = @view layer_idxs[1:n_total]
        proposal = MultiSpinProposal(idxs_view, from_view, to_view, layer_view, accept_move)

        if accept_move
            copyto!(visible, new_v)
            copyto!(hidden, new_h)
            update!(algorithm, hamiltonian, model, proposal)
            accepted += 1
        end
    end

    acceptance_rate = attempted == 0 ? zero(SType) : SType(accepted) / SType(attempted)
    return (;
        proposal,
        accepted,
        attempted,
        acceptance_rate,
        log_acceptance,
        η = α0,
        group_steps = n_group_steps,
        langevin_steps = K,
        adjusted = adjusted[],
        adjust_step = adjust_step[],
        burned_in = step_idx[] > burnin[],
        step_idx = step_idx[],
    )
end
