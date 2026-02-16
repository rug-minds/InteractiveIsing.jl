struct MetropolisOLD <: MCAlgorithm end

export MetropolisOLD
requires(::Type{MetropolisOLD}) = Δi_H()

function reserved_symbols(::Type{MetropolisOLD})
    return [:w_ij => :wij, :sn_i => :newstate, :s_i => :oldstate, :s_j => :(gstate[j])]
end

function collect_ex(gstate, gadj, i)
    cumsum = zero(eltype(gstate))
    @turbo for ptr in nzrange(gadj, i)
        j = gadj.rowval[ptr]
        wij = gadj.nzval[ptr]
        cumsum += wij * gstate[j]
    end
    return cumsum
end

function example_ising(i, gstate, newstate, oldstate, gadj, gparams, lt)
    cumsum = zero(eltype(gstate))
    @turbo for ptr in nzrange(gadj, i)
        j = gadj.rowval[ptr]
        wij = gadj.nzval[ptr]
        cumsum += wij * gstate[j]
    end
    return (oldstate-newstate) * cumsum
end

function Processes.init(::MetropolisOLD, args::As) where As
    (;g) = args
    gstate = g.state
    gadj = g.adj
    type = eltype(gstate)
    len = length(gstate)
    gparams = Parameters(self = ParamTensor(zeros(type, len), 0, "Self Connections", false), b = ParamTensor(zeros(type, len), 0, "Magnetic Field", false))
    iterator = ising_it(g)
    # rng = MersenneTwister()
    rng = Random.GLOBAL_RNG
    extraargs = Hamiltonian_Builder(MetropolisOLD, g, gparams, IsingOLD())
    M = Ref(sum(g.state))
    lmeta = LayerMetaData(g[1])
    return (;gstate, gadj, gparams, iterator, lmeta, rng, M, extraargs...)
end

@inline function (::MetropolisOLD)(@specialize(args))
    #Define vars
    (;g, gstate, gadj, gparams, iterator, ΔH, lmeta, rng, M) = args
    i = rand(rng, iterator)
    MetropolisOLD(args, i, g, gstate, gadj, gparams, M, ΔH, rng, lmeta)
end

@inline function MetropolisOLD(args, i, g, gstate::Vector{T}, gadj, gparams, M, ΔH, rng, lmeta) where {T}
    β = one(T)/(temp(g))
    
    oldstate = @inbounds gstate[i]
    
    newstate = @inline sampleState(statetype(lmeta), oldstate, rng, stateset(lmeta))   

    ΔE = @inline ΔH(i, gstate, newstate, oldstate, gadj, gparams, lmeta)
    
    efac = exp(-β*ΔE)

    if (ΔE <= zero(T) || rand(rng, T) < efac)
        @inbounds gstate[i] = newstate 
        @hasarg if M isa Ref
            M[] += (newstate - oldstate)
        end
    end
    
    return nothing
end