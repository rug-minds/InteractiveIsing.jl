export LayeredMetropolis

struct LayeredMetropolis <: MCAlgorithm end
# requires(::Type{LayeredMetropolis}) = Δi_H()
# requires(::LayeredMetropolis) = deltH()


# function reserved_symbols(::Type{LayeredMetropolis})
#     return [:w_ij => :wij, :sn_i => :newstate, :s_i => :oldstate, :s_j => :(gstate[j])]
# end

function Processes.prepare(::LayeredMetropolis, @specialize(args))
    (;g) = args
    args = prepare(Metropolis(), args)
    Ms = [Ref(sum(state(g[i]))) for i in 1:length(g.layers)]

    return (;args..., 
            layerarch = GetArchitecture(unshuffled(g.layers)...),
            Ms = Ms)
end


Base.@propagate_inbounds @inline function (::LayeredMetropolis)(@specialize(args))
    #Define varso
    
    (;iterator, rng, layerarch) = args
    j = rand(rng, iterator)
    @inline layerswitch(LayeredMetropolis, j, layerarch, args)
end

Base.@propagate_inbounds @inline function LayeredMetropolis(j, args, layeridx, lmeta)
    M = args.Ms[layeridx]
    args = (;args..., j, layeridx, M, lmeta)
    @inline Metropolis()(args)
end