export LayeredMetropolis

# struct LayeredMetropolis <: MCAlgorithm end
# requires(::Type{LayeredMetropolis}) = Δi_H()
# requires(::LayeredMetropolis) = deltH()


# function reserved_symbols(::Type{LayeredMetropolis})
#     return [:w_ij => :wij, :sn_i => :newstate, :s_i => :oldstate, :s_j => :(gstate[j])]
# end

@ProcessAlgorithm function LayeredMetropolis(iterator, rng, layers, context)
    #Define varso
    
    # (;iterator, rng, layerarch) = context
    j = rand(rng, iterator)
    @inline layerswitch(Layered_step, j, layers, context)
end

function Processes.init(::LayeredMetropolis, context::Con) where Con
    (;g) = context
    context = init(Metropolis(), context)
    Ms = [Ref(sum(state(g[i]))) for i in 1:length(g.layers)]

    return (;context..., 
            layers = g.layers,
            Ms = Ms)
end

Base.@propagate_inbounds @inline function Layered_step(j, context::NamedTuple, layeridx, layer)
    M = context.Ms[layeridx]
    args = (;context..., j, layeridx, M, layer)
    @inline Metropolis_step(context, args)
end