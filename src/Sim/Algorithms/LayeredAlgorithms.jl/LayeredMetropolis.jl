function layeredMetropolis end

function get_args(::typeof(layeredMetropolis))
    return (:g, :gstate, :gadj, :gparams, :iterator, :rng, :gstype, :ΔEFunc)
end


"""
Should take in kwargs and return a namedtuple that has all variables the algorithm needs
"""
function Processes.init(::typeof(layeredMetropolis), g, prt = false; kwargs...)::NamedTuple
    l_iterators = Val{tuple(layeridxs(g)...)}()
    def_kwargs = pairs((;g,
                        gstate = state(g),
                        gadj = adj(g),
                        gparams = params(g),
                        iterator = ising_it(g),
                        rng = MersenneTwister(),
                        gstype = stype(g),
                        params = g.params,
                        ΔEFunc = ΔEIsing,
                        layers = layers(g).data,
                        l_iterators = l_iterators,
                    ))

    changeable_kwargs = pairs((;))
    ## Extremise states discrete states
    ## TODO: Only do this when switching from different algorithm
    extremiseDiscrete!.(layers(g))
    # changeable_kwargs = replacekwargs(changeable_kwargs, kwargs)
    newargs = (;mergekwargs(def_kwargs, kwargs)...)
    return newargs
end

@generated function layeredMetropolis(@specialize(args))
    expr = Meta.parse(gen_exp(layeredMetropolis, args))
    return expr
end

@inline function updateMetropolisLayered(idx::Integer, g::IsingGraph{T}, gstate, gadj, rng, gstype::SimT, ΔEFunc, ::Type{StateT}, StateSet) where {T, SimT <: SType, StateT <: StateType}
    β = one(T)/(temp(g))
    
    oldstate = @inbounds gstate[idx]
    
    newstate = sampleState(StateT, oldstate, rng, StateSet)

    ΔE = ΔEFunc(g, oldstate, newstate, gstate, gadj, idx, gstype, StateT)
    efac = exp(-β*ΔE)
    randnum = rand(rng, Float32)
    if (ΔE <= zero(T) || randnum < efac)
        @inbounds gstate[idx] = newstate 
    end
    return nothing
end


function gen_exp(::typeof(layeredMetropolis), argstype)

    # Unpack the args tuple
    expr_str = "begin 
        $(get_args_string(layeredMetropolis)) = args 
        idx = rand(rng, iterator)
        $(generate_layerswitch(layeredMetropolis, grouped_ltypes_idxs(argstype)))
end"

    return expr_str
end

# TODO: Make this more general, such that other algorithms can be plugged in
function generate_layerswitch(::typeof(layeredMetropolis), typeidx_zip)
    startstr = ""
    statements = 1
    for (layertype, gidxs) in typeidx_zip
        _statetype = statetype(layertype)
        if _statetype == Static
            continue
        end
        _stateset = stateset(layertype)
        if statements == 1
            startstr *= "if"
        else
            startstr *= "\telseif"
        end

        startstr *= " idx <= $(gidxs[end])\n"
        startstr *= "\t\tupdateMetropolisLayered(idx, g, gstate, gadj, rng, gstype, ΔEFunc, $(_statetype), $(_stateset))\n"
        statements += 1
    end
    startstr *= "\tend"
    return startstr
end

export init

export layeredMetropolis




