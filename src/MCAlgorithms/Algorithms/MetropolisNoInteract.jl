export metropolis_nointeract

struct metropolis_nointeract{TT<:Real} <: IsingMCAlgorithm
    T::TT
end

metropolis_nointeract(; T = 1.0f0) = metropolis_nointeract(T)

@inline function Processes.init(metro::metropolis_nointeract, context::Cont) where {Cont}
    (;model) = context

    hamiltonian = model.hamiltonian
    rng = Random.MersenneTwister()

    hamiltonian = init!(hamiltonian, model)
    proposer = get_proposer(model)

    return (;model, hamiltonian, proposer, rng)
end

@inline function Processes.step!(metro::metropolis_nointeract, context::C) where {C}
    (;rng, model, hamiltonian, proposer) = context

    proposal = @inline rand(rng, proposer)
    Ttype = eltype(model)
    T = Ttype(metro.T)

    ΔE = @inline calculate(ΔH(), hamiltonian, model, proposal)
    if (@inline (ΔE <= zero(Ttype) || rand(rng, Ttype) < exp(-ΔE / T)))
        proposal = @inline accept(proposer, proposal)
    end

    @inline update!(Metropolis(), hamiltonian, model, proposal)
    return (;proposal, ΔE, T)
end
