# AI Generated

export d_sH, pump

"""
    d_sH()

Whole-state Hamiltonian derivative functional, representing `dH / ds` for the
current state used by an adiabatic optimizer.
"""
struct d_sH <: AbstractLinearFunctional end

"""
    pump(hterm, model, step, total_steps; backend = CPUBackend())

Return the schedule multiplier for one Hamiltonian term during adiabatic
derivative accumulation.
"""
function pump(
    hterm::H,
    model::M,
    step::I,
    total_steps::J;
    backend::B = CPUBackend(),
) where {H<:Hamiltonian,M<:AbstractIsingGraph,I<:Integer,J<:Integer,B<:OptimizationBackend}
    return one(eltype(model))
end

"""
    calculate(d_sH(), hamiltonian, model; backend = CPUBackend(), step = 0, total_steps = 1)

Allocate and return the whole-state Hamiltonian derivative for `hamiltonian`.
"""
function calculate(
    ds::d_sH,
    hamiltonian::H,
    model::M;
    backend::B = CPUBackend(),
    step::Integer = 0,
    total_steps::Integer = 1,
) where {H,M<:AbstractIsingGraph,B<:OptimizationBackend}
    dest = similar(backend_state(backend, model))
    calculate!(dest, ds, hamiltonian, model; backend, step, total_steps)
    return dest
end

"""
    calculate!(dest, d_sH(), hamiltonian, model; backend = CPUBackend(), step = 0, total_steps = 1)

Overwrite `dest` with the whole-state Hamiltonian derivative for `hamiltonian`.
"""
function calculate!(
    dest::AbstractVector,
    ds::d_sH,
    hamiltonian::H,
    model::M;
    backend::B = CPUBackend(),
    step::Integer = 0,
    total_steps::Integer = 1,
) where {H<:Hamiltonian,M<:AbstractIsingGraph,B<:OptimizationBackend}
    fill!(dest, zero(eltype(dest)))
    scale = pump(hamiltonian, model, step, total_steps; backend)
    _accumulate_whole_state_derivative!(dest, ds, hamiltonian, model, backend, scale)
    return dest
end

"""
    calculate!(dest, d_sH(), hamiltonian_terms, model; backend = CPUBackend(), step = 0, total_steps = 1)

Overwrite `dest` with the summed whole-state derivative of all Hamiltonian
terms in `hamiltonian_terms`.
"""
function calculate!(
    dest::AbstractVector,
    ds::d_sH,
    hamiltonian_terms::HTS,
    model::M;
    backend::B = CPUBackend(),
    step::Integer = 0,
    total_steps::Integer = 1,
) where {HTS<:AbstractHamiltonianTerms,M<:AbstractIsingGraph,B<:OptimizationBackend}
    fill!(dest, zero(eltype(dest)))
    for hamiltonian in hamiltonians(hamiltonian_terms)
        scale = pump(hamiltonian, model, step, total_steps; backend)
        _accumulate_whole_state_derivative!(dest, ds, hamiltonian, model, backend, scale)
    end
    return dest
end

"""
    _accumulate_whole_state_derivative!(dest, d_sH(), hterm, model, backend, scale)

Accumulate one Hamiltonian term's whole-state derivative into `dest`.
"""
function _accumulate_whole_state_derivative!(
    dest::AbstractVector,
    ds::d_sH,
    hterm::H,
    model::M,
    backend::CPUBackend,
    scale::S,
) where {H<:Hamiltonian,M<:AbstractIsingGraph,S<:Real}
    spins = backend_state(backend, model)
    model_spins = graphstate(model)
    T = eltype(dest)

    # Reuse scalar derivatives on CPU, presenting an active CPU x-buffer as graph state.
    restore_state = !(spins === model_spins)
    original_spins = restore_state ? collect(model_spins) : nothing
    if restore_state
        copyto!(model_spins, collect(spins))
    end

    try
        for layer in layers(model)
            for spin_idx in graphidxs(layer)
                proposal = SingleSpinProposal{eltype(model)}(
                    spin_idx,
                    spins[spin_idx],
                    NoChange(),
                    layeridx(layer),
                    false,
                )
                @inbounds dest[spin_idx] += T(scale) * T(calculate(d_iH(), hterm, model, proposal))
            end
        end
    finally
        if restore_state
            copyto!(model_spins, original_spins)
        end
    end

    return dest
end

"""
    _accumulate_whole_state_derivative!(dest, d_sH(), hterm, model, backend, scale)

Reject GPU derivative execution for terms without an explicit backend
implementation.
"""
function _accumulate_whole_state_derivative!(
    dest::AbstractVector,
    ds::d_sH,
    hterm::H,
    model::M,
    backend::B,
    scale::S,
) where {H<:Hamiltonian,M<:AbstractIsingGraph,B<:AbstractGPUBackend,S<:Real}
    throw(ArgumentError("Hamiltonian term $(typeof(hterm)) does not implement calculate!(dest, d_sH(), ...; backend=$(typeof(backend)))."))
end
