export Metropolis
struct Metropolis <: IsingMCAlgorithm end


@inline function IsingMetropolis()
    return Unique(Metropolis())
    # destr = DestructureInput()
    # metro = Metropolis(tracked = tracked)
    # package(CompositeAlgorithm(metro, destr, Route(destr => metro, :isinggraph => :structure)))
end

@inline MetropolisTracked() = Metropolis()

@inline function Processes.init(::Metropolis, context::Cont) where {Cont}
    # @show typeof(context)
    (;model) = context

    hamiltonian = model.hamiltonian

    rng = Random.MersenneTwister()

    hamiltonian = init!(hamiltonian, model)
    proposer = get_proposer(model)

    T = eltype(model)(temp(model))
    returnargs = (;model, hamiltonian, proposer, rng, T)
    return returnargs
end

# @inline function (::Metropolis)(context::As) where As
@inline function Processes.step!(metro::Metropolis, context::C) where {C}
    (;rng, model, hamiltonian, proposer, T) = context

    floattype = eltype(model)
    proposal = @inline rand(rng, proposer)

   
    ΔE = @inline calculate(ΔH(), hamiltonian, model, proposal)
    if (@inline (ΔE <= zero(floattype) || rand(rng, floattype) < exp(-ΔE / T)))
        proposal = @inline accept(proposer, proposal)
    end

    @inline update!(metro, hamiltonian, model, proposal)
    return (;proposal, ΔE)
end
