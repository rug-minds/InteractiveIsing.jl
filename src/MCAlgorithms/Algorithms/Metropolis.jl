export Metropolis

struct Metropolis <: MCAlgorithm end
struct MetropolisNew <: MCAlgorithm end
struct deltaH end

export MetropolisNew
requires(::Type{Metropolis}) = Δi_H()

function reserved_symbols(::Type{Metropolis})
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

function Processes.prepare(::Metropolis, @specialize(args))
    (;g) = args
    gstate = g.state
    gadj = g.adj
    gparams = g.params
    iterator = ising_it(g)
    # rng = MersenneTwister()
    rng = Random.GLOBAL_RNG
    ΔH = Hamiltonian_Builder(Metropolis, g, Ising())
    M = Ref(sum(g.state))
    lmeta = LayerMetaData(g[1])
    return (;gstate, gadj, gparams, iterator, ΔH, lmeta, rng, M)
end

@inline function (::Metropolis)(@specialize(args))
    #Define vars
    (;g, gstate, gadj, gparams, iterator, ΔH, lmeta, rng, M) = args
    i = rand(rng, iterator)
    Metropolis(args, i, g, gstate, gadj, gparams, M, ΔH, rng, lmeta)
end

@inline function Metropolis(args, i, g, gstate::Vector{T}, gadj, gparams, M, ΔH, rng, lmeta) where {T}
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

function Processes.prepare(::MetropolisNew, @specialize(args))
    (;g) = args
    gstate = g.state
    gadj = g.adj
    params = g.params
    iterator = ising_it(g)
    hamiltonian = init!(g.hamiltonian, g)
    deltafunc = deltaH(hamiltonian)
    rng = Random.GLOBAL_RNG
    M = Ref(sum(g.state))
    Δs_i = Ref(zero(eltype(g.state)))

    lmeta = LayerMetaData(g[1])
    return (;g, gstate, gadj, params, iterator, hamiltonian, deltafunc, lmeta, rng, M, Δs_i)
end

@inline function (::MetropolisNew)(@specialize(args))
    #Define vars
    (;iterator, rng) = args
    j = rand(rng, iterator)
    MetropolisNew((;args..., j))
end

@inline function MetropolisNew(args::As) where As
    (;g, gstate, j, deltafunc, M, rng, lmeta) = args
    T = eltype(g)
    β = one(T)/(temp(g))
    oldstate = @inbounds gstate[j]
 
    newstate = SparseVal((@inline sampleState(statetype(lmeta), oldstate, rng, stateset(lmeta)))::eltype(gstate), Int32(length(gstate)), j)::SparseVal{eltype(gstate), Int32}

    ΔE = @inline deltafunc((;args..., newstate); j)
    
    efac = exp(-β*ΔE)
    if (ΔE <= zero(T) || rand(rng, T) < efac)
        @inbounds gstate[j] = newstate 
        @hasarg if M isa Ref
            M[] += (newstate - oldstate)
        end
        @hasarg if Δs_i isa Ref
            Δs_i[] = newstate - oldstate
        end
    else
        @hasarg if Δs_i isa Ref
            Δs_i[] = 0
        end
    end

    @inline update!(args.hamiltonian, args)
    return nothing
end

@inline specific_hamkw(@specialize(args); @specialize(kwargs...)) = @inline specific_ham(args, (;kwargs...))

function specific_ham(@specialize(args), @specialize(kwargs))
    (;params) = args
    (;j, newstate) = kwargs
    adj = args.gadj
    state = args.gstate
    cumsum = zero(Float32)
    for ptr in nzrange(adj, j)
        i = adj.rowval[ptr]
        wij = adj.nzval[ptr]
        cumsum += wij * state[i]
    end
    return (state[j]-newstate) * cumsum + (state[j]^2-newstate^2)*params.self[j] + (state[j]-newstate)*params.b[j]
    # return (state[j]-newstate) * cumsum + (state[j]-newstate)*params.b[j] 
end

function specific_ham(@specialize(args), newstate, j)
    (;params) = args
    adj = args.gadj
    state = args.gstate
    cumsum = zero(Float32)
    for ptr in nzrange(adj, j)
        i = adj.rowval[ptr]
        wij = adj.nzval[ptr]
        cumsum += wij * state[i]
    end
    return (state[j]-newstate) * cumsum + (state[j]^2-newstate^2)*params.self[j] + (state[j]-newstate)*params.b[j]
end




