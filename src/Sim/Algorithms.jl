
"""
Fallback preparation for updateFunc
"""
prepare(::Any, ::Any; kwargs...) = error("No prepare function defined for $(typeof(algorithm))")

@inline sampleState(oldstate, ::Type{Discrete}, rng) = -oldstate
@inline sampleState(oldstate, ::Type{Continuous}, rng) = rand(Uniform(-1f0, 1f0))

@inline function updateMetropolis(@specialize(args))
    #Define vars
    (;g, gstate, gadj, gparams, iterator, rng, gstype, ΔEFunc) = args
    idx = rand(rng, iterator)
    updateMetropolis(idx, g, gstate, gadj, rng, gstype, ΔEFunc)
end


@inline function updateMetropolis(idx::Integer, g, gstate::Vector{Int8}, gadj, rng, gstype::ST, ΔEFunc) where {ST <: SType}

    β = 1f0/(temp(g))

    oldstate = @inbounds gstate[idx]
    
    ΔE = ΔEFunc(g, oldstate, 1, gstate, gadj, idx, gstype, Discrete)

    if (ΔE <= 0f0 || rand(rng, Float32) < exp(-β*ΔE))
        @inbounds gstate[idx] *= -Int8(1)
    end
    return nothing
end


@inline function updateMetropolis(idx::Integer, g, gstate::Vector{Float32}, gadj, rng, gstype::ST, ΔEFunc) where {ST <: SType}

    β = 1f0/(temp(g))

    oldstate = @inbounds gstate[idx]

    newstate = 2f0*(rand(rng, Float32)- .5f0)

    ΔE = ΔEFunc(g, oldstate, newstate, gstate, gadj, idx, gstype, Continuous)

    if (ΔE < 0f0 || rand(rng, Float32) < exp(-β*ΔE))
        @inbounds g.state[idx] = newstate 
    end

    return nothing
end

function get_args(::typeof(updateMetropolis))
    return (:g, :gstate, :gadj, :iterator, :rng, :gstype, :ΔEFunc)
end

@inline function ΔEIsing(g, oldstate, newstate, gstate, gadj, idx, @specialize(gstype), ::Type{Discrete})
    # println("Discrete algo called on idx $idx")
    # println("Calculated energy $(@inbounds 2f0*oldstate*dEIsing(g, gstate, gadj, idx, gstype))")
    # sleep(0.5)
    return @inbounds -2f0*oldstate*dEIsing(g, gstate, gadj, idx, gstype)
end
@inline function ΔEIsing(g, oldstate, newstate, gstate, gadj, idx, @specialize(gstype), ::Type{Continuous})
    efactor = dEIsing(g, gstate, gadj, idx, gstype)
    # return EdiffIsing(g, gstype, idx, efactor, oldstate, newstate)
    return @inbounds efactor*(newstate-oldstate)
end

function prepare(::typeof(updateMetropolis), g; kwargs...)
    def_kwargs = pairs((;g,
                        gstate = state(g),
                        gadj = sp_adj(g),
                        gparams = params(g),
                        iterator = ising_it(g, stype(g)),
                        rng = MersenneTwister(),
                        gstype = stype(g),
                        ΔEFunc = ΔEIsing,
                    ))
    return (;replacekwargs(def_kwargs, kwargs)...)
end

export updateMetropolis



using Distributions: Normal
const stepsize = Ref(0.01f0)
setstepsize(val) = stepsize[] = val
export setstepsize

const langevin_prealloc  = Float32[]
function updateLangevinThreaded(g::IsingGraph, gstate, gadj, iterator, rng, gstype::ST, dEFunc) where {ST <: SType}
    Threads.@threads for (i_idx, s_idx) in collect(enumerate(iterator))
        langevin_prealloc[i_idx] = dEFunc(g, gstate, gadj, s_idx, gstype)
    end
    grad = @view langevin_prealloc[1:length(iterator)]
    noise = rand(Normal(0f0,1f0), length(iterator))
    @inbounds (@view gstate[iterator]) .= clamp!((@view gstate[iterator]) - (stepsize[])*grad + sqrt(2f0*stepsize[]*temp(g))*noise, -1f0, 1f0)
end

function updateLangevin(g::IsingGraph, gstate, gadj, iterator, rng, gstype::ST, dEFunc) where {ST <: SType}
    for (i_idx, s_idx) in enumerate(iterator)
        langevin_prealloc[i_idx] = dEFunc(g, gstate, gadj, s_idx, gstype)
    end
    grad = @view langevin_prealloc[1:length(iterator)]
    noise = rand(Normal(0f0,1f0), length(iterator))
    @inbounds (@view gstate[iterator]) .= clamp!((@view gstate[iterator]) - (stepsize[])*grad + sqrt(2f0*stepsize[]*temp(g))*noise, -1f0, 1f0)
end
function prepare(::Union{typeof(updateLangevin), typeof(updateLangevinThreaded)}, g; kwargs...)
    resize!(langevin_prealloc, length(state(g)))
end
export updateLangevin

let times = Ref([])
    global function upDebug(g, params, lTemp, gstate::Vector, gadj, iterator, rng, gstype, dEFunc)

        β = 1/(lTemp[])
        
        idx = rand(rng, iterator)
        
        ti = time()
        Estate = @inbounds gstate[idx]*dEFunc(g, gstate, gadj, idx, gstype)
        tf = time()

        push!(times[], tf-ti)
        if length(times[]) == 1000000
            println(sum(times[])/length(times[]))
            times[] = []
        end

        minEdiff = 2*Estate

        if (Estate >= 0 || rand(rng) < exp(β*minEdiff))
            @inbounds g.state[idx] *= -1
        end
        
    end
end
export upDebug

abstract type ΔE end
abstract type dE end
(ΔE)(::typeof(ΔEIsing)) = true
(dE)(::typeof(dEIsing)) = true


## GAUSSIAN BERNOULLI

function ΔE_GB(g::IsingGraph{Float32}, params, oldstate, newstate, gstate, gadj, idx, gstype)
    return nothing
end

function get_params(::typeof(ΔE_GB))
    return (;σ = Vector{Float32}, μ = Vector{Float32}, b = Vector{Float32}, bare_adj = SparseMatrixCSC{Float32,Int32})
end

function generate_layercode(statements, iterator, ltype::Type{IsingLayer{A,B}}) where {A,B}
    if statements == 1
        startstr = "if"
    else
        startstr = "elseif"
    end
    startstr *= " idx <= $(iterator[end])\n"
    startstr *= "\tupdateMetropolisLayered(idx, g, gstate, gadj, rng, gstype, ΔEFunc, $(A))\n"
end

function _layeredMetropolis_exp(argstype)


    # Unpack the args tuple
    expr_str = "begin "*get_args_string(layeredMetropolis)*" = args\n"
    expr_str *= "idx = rand(rng, iterator)\n"

    layertypes = gettype(argstype, :layers).parameters
    layeridxs = getval(gettype(argstype, :l_iterators))

    statements = 1
    for idx in eachindex(layertypes)
        current_layertype = layertypes[idx]
        if idx != length(layertypes)
            next_layertype = layertypes[idx+1]
            if current_layertype == next_layertype
                continue
            end
        end
        expr_str *= generate_layercode(statements, layeridxs[idx], current_layertype)
        statements += 1
    end

    expr_str *= "end end"

    return expr_str
end

@generated function layeredMetropolis(@specialize(args))
    expr = Meta.parse(_layeredMetropolis_exp(args))
    return expr
end

function prepare(::typeof(layeredMetropolis), g; kwargs...)
    l_iterators = Val{tuple(layeridxs(g)...)}()
    def_kwargs = pairs((;g,
                        gstate = state(g),
                        gadj = sp_adj(g),
                        gparams = params(g),
                        iterator = ising_it(g, stype(g)),
                        rng = MersenneTwister(),
                        gstype = stype(g),
                        ΔEFunc = ΔEIsing,
                        layers = layers(g).data,
                        l_iterators = l_iterators,
                    ))
    ## Extremise states
    ## TODO: Only do this when switching from different algorithm
    ## thus the process should track which algorithm it is using
    for layer in layers(g)
        if statetype(layer) == Discrete
            state(layer) .= sign.(state(layer))
        end
    end
    return (;replacekwargs(def_kwargs, kwargs)...)
end

export layeredMetropolis

@inline function updateMetropolisLayered(idx::Integer, g, gstate, gadj, rng, gstype::SimT, ΔEFunc, ::Type{StateT}) where {SimT <: SType, StateT <: StateType}
    β = 1f0/(temp(g))
    
    oldstate = @inbounds gstate[idx]
    newstate = sampleState(oldstate, StateT, rng)
    
    ΔE = ΔEFunc(g, oldstate, newstate, gstate, gadj, idx, gstype, StateT)
    efac = exp(-β*ΔE)
    randnum = rand(rng, Float32)
    if (ΔE < 0f0 || randnum < efac)
        @inbounds gstate[idx] = newstate 
    end
    return nothing
end



function get_args(::typeof(layeredMetropolis))
    return (:g, :gstate, :gadj, :params, :iterator, :rng, :gstype, :ΔEFunc)
end

function get_args_string(func_type, addbrackets = nothing)
    args = "$(get_args(func_type))"
    # Remove colons
    args = replace(args, ":" => "")
    if !isnothing(addbrackets)
        args = args[1:end-1]
        args *= ", $addbrackets)"
    end
    return args
end