
export overlayNoise!, resetstate!, activateone!
# Setting elements

"""
Set spins either to a value or clamp them
"""
#Clean this up
# TODO: Shouldn't always reinit sim if the iterator didn't change
function setSpins!(g::AbstractIsingGraph{T}, idxs::Union{AbstractArray{<:Integer}, AbstractArray{<:Integer}, <:UnitRange, AbstractArray{<:CartesianIndex}}, val, clamp::Bool = false) where T
    hasdefects_before = hasDefects(graph(g))

    # Set the defects
    clamprange!(g, clamp, idxs)

    hasdefects_after = hasDefects(graph(g))
    if hasdefects_before != hasdefects_after
        reinit(graph(g))
    end

    # Set the spins
    copystate(g, idxs, val)
end

function setdefect!(g::AbstractIsingLayer{T,D}, vals, idxs) where {T,D}

end

copystate(g::AbstractIsingGraph, idxs, val::Real) = @inbounds state(g)[idxs] .= closestTo(g, val)

copystate(g::AbstractIsingGraph, idxs, val::AbstractArray) = @inbounds state(g)[idxs] .= val[1:end]

copystate(g::IsingLayer, idxs, val::AbstractArray) = clamp_to_stateset(g, (@view state(g)[idxs]), val)



setSpins!(g::AbstractIsingGraph, vals::AbstractArray, clamp::Bool = false) = setSpins!(g, graphidxs(g), vals, clamp)

function setSpins!(g::AbstractIsingGraph, coords::Union{Vector{NTuple{3,Int}}, Vector{NTuple{2,Int}}}, val::Real, clamp::Bool = false)
    setSpins!(g, coordToIdx.(coords, Ref(size(g))), val, clamp)
end

function setSpins!(g::AbstractIsingGraph{T}, idx::Integer, val::Real, clamp::Bool = false) where T
    hasdefects_before = hasDefects(graph(g))
    
    setdefect(g, clamp, idx)

    hasdefects_after = hasDefects(graph(g))

    if hasdefects_before != hasdefects_after
        reinit(graph(g))
    end

    @inbounds state(g)[idx] = val
end

function setDefects!(g, val, idxs)
    hasdefects_before = hasDefects(graph(g))
    
    index_set(g)[idxs] = val
    
    hasdefects_after = hasDefects(graph(g))
    
    if hasdefects_before != hasdefects_after
        reinit(graph(g))
    end

    return idxs
end
export setDefects!, resetDefects!

function resetDefects!(g::AbstractIsingGraph)
    g = graph(g)

    hasdefects_before = hasDefects(g)
    setDefects!(g, false, graphidxs(g))
    hasdefects_after = hasDefects(g)
    if hasdefects_before != hasdefects_after
        reinit(g)
    end
end

function clampImg!(layer::IsingLayer, imgfile)
    # Load the image
    img = load(imgfile)

    # Resize the image
    img = imresize(img, (Int64(glength(layer)), Int64(gwidth(layer))))

    # # Convert to black and white image
    img = Gray.(img)
    img = img .> 0.5
    img = img .*2 .- 1    

    setSpins!(layer, [1:length(img);], (permutedims(img)[:,end:-1:1])[:] , true)

    return

end

clampImg!(g, layeridx::Integer, imgfile) = clampImg!(layers(g)[layeridx], imgfile)
export clampImg!

function copyState!(layer1, layer2, clamp = false)
    imresize(state(layer1), (Int64(glength(layer2)), Int64(gwidth(layer2))))
    state(layer2) .= state(layer1)
end
copyState!(g, layeridx1::Integer, layeridx2::Integer, clamp = false) = copyState!(layer(g, layeridx1), layer(g, layeridx2), clamp)
export copyState!

function overlayNoise!(layer::IsingLayer, p; noise_values = [-1, 1])
    maskVec = rand(length(state(layer))) .< (p/100)
    idxs = [i for (i, x) in enumerate(maskVec) if x]
    states = rand(noise_values, length(idxs))
    state(layer)[idxs] .= states
end

overlayNoise!(g, layeridx::Integer, p; noise_values = [-1, 1]) = overlayNoise!(layers(g)[layeridx], p; noise_values)
resetstate!(g::IsingGraph) = graphstate(g) .= initRandomState(g)
resetstate!(l::IsingLayer) = state(l)[:] .= rand(l, length(state(l)))
#TODO: This is a shitty implementation
resetstate!(layers::IsingLayer...) = for l in layers; resetstate!(l); end

"""
For a layer, set all to zero and 1 to 1
"""
activateone!(l::IsingLayer, idx, val = 1, allval = 0) = begin state(l) .= allval; state(l)[idx] = val end
"""
    _process_temperature_value(value)

Return the numeric payload for process-context temperature storage, or
`nothing` when the slot is not a supported temperature value.
"""
@inline _process_temperature_value(value::T) where {T<:Real} = value
@inline _process_temperature_value(value::Base.RefValue{T}) where {T<:Real} = value[]
@inline _process_temperature_value(value::StatefulAlgorithms.InteractiveVar{T}) where {T<:Real} = value[]
@inline _process_temperature_value(value::StatefulAlgorithms.InteractiveVar{<:Base.RefValue{T}}) where {T<:Real} = value[][]
@inline _process_temperature_value(_) = nothing

"""
    _process_temperature_vars(subcontext)

Collect the process-context variables that represent Monte Carlo temperature.
Both `:temp` and `:T` are accepted for compatibility with existing algorithms.
"""
function _process_temperature_vars(subcontext)
    data = StatefulAlgorithms.getdata(subcontext)
    pairs = Pair{Symbol, Any}[]
    for name in (:temp, :T)
        haskey(data, name) || continue
        value = getproperty(data, name)
        isnothing(_process_temperature_value(value)) || push!(pairs, name => value)
    end
    return pairs
end

"""
    _set_process_temperature_slot!(slot, value)

Write a converted internal temperature into mutable process-context storage.
Returns `nothing` when the slot is immutable and must be handled by rebuilding
or interacting with the process context instead.
"""
function _set_process_temperature_slot!(slot::Base.RefValue{T}, value) where {T<:Real}
    slot[] = convert(T, value)
    return slot[]
end

function _set_process_temperature_slot!(slot::StatefulAlgorithms.InteractiveVar{T}, value) where {T<:Real}
    slot[] = convert(T, value)
    return slot[]
end

function _set_process_temperature_slot!(slot::StatefulAlgorithms.InteractiveVar{<:Base.RefValue{T}}, value) where {T<:Real}
    slot[][] = convert(T, value)
    return slot[][]
end

@inline _set_process_temperature_slot!(slot, value) = nothing

"""
    _set_process_context_temperature!(process, value)

Propagate one converted internal graph temperature into the modern process-list
state without using the deprecated graph `sim` slot.
"""
function _set_process_context_temperature!(process::P, value) where {P<:StatefulAlgorithms.AbstractProcess}
    context = StatefulAlgorithms.context(process)
    context isa StatefulAlgorithms.ProcessContext || return nothing

    # Existing algorithms use either `:temp` or `:T`; update all visible
    # subcontexts while skipping framework-owned bookkeeping contexts.
    subcontexts = StatefulAlgorithms.get_subcontexts(context)
    for subcontext_name in propertynames(subcontexts)
        subcontext_name === :globals && continue
        subcontext_name === :_injector && continue
        subcontext_name === :_exchange && continue

        subcontext = getproperty(subcontexts, subcontext_name)
        for (varname, current) in _process_temperature_vars(subcontext)
            if !isnothing(_set_process_temperature_slot!(current, value))
                continue
            elseif StatefulAlgorithms.isinteractive(process)
                StatefulAlgorithms.interact!(process, varname => value)
            elseif !StatefulAlgorithms.isrunning(process)
                converted = convert(typeof(current), value)
                update = NamedTuple{(subcontext_name,)}((NamedTuple{(varname,)}((converted,)),))
                StatefulAlgorithms.context(process, StatefulAlgorithms.merge_into_subcontexts(context, update))
            end
        end
    end
    return nothing
end

"""
    settemp!(g, value)

Set graph temperature, converting Unitful inputs through `temp!`, then mirror
the converted internal value into any attached process contexts.
"""
function settemp!(g::G, value) where {G<:IsingGraph}
    temp!(g, value)
    converted = temp(g)
    for process in processes(g)
        _set_process_context_temperature!(process, converted)
    end
    return converted
end
export settemp!

function _physical_parameter_units(term, param::Symbol)
    params = parameters(term)
    if params isa Parameters
        return get(getfield(params, :units), param, nothing)
    elseif term isa GaussianBernoulli && param in (:w, :W, :μ, :mu, :logσ2, :logsigma2, :b)
        return physicalunits(role = :dimensionless)
    else
        return nothing
    end
end

function _assign_physical_value!(storage::Base.RefValue, converted, idxs)
    storage[] = converted
    return storage
end

function _assign_physical_value!(storage::AbstractArray, converted, idxs)
    if Base.ndims(storage) == 0
        storage[] = converted
    elseif idxs isa Colon
        converted isa AbstractArray ? (storage .= converted) : fill!(storage, converted)
    else
        converted isa AbstractArray ? (storage[idxs] .= converted) : (storage[idxs] .= converted)
    end
    return storage
end

function _assign_physical_value!(storage, converted, idxs)
    throw(ArgumentError("Hamiltonian parameter storage $(typeof(storage)) is not mutable through `setphysical!`. Use mutable storage such as `UniformArray`, `Vector`, or `Ref`."))
end

"""
    setphysical!(model, term_type, param, value; idxs = :)

Convert a physical value with the parameter's unit metadata and write the
result into the instantiated Hamiltonian storage.
"""
function setphysical!(model::M, ::Type{H}, param::Symbol, value; idxs = Colon()) where {M<:AbstractIsingGraph,H<:Hamiltonian}
    term = gethamiltonian(hamiltonian(model), H)
    return setphysical!(model, term, param, value; idxs)
end

function setphysical!(layer::L, ::Type{H}, param::Symbol, value; idxs = Colon()) where {L<:AbstractIsingLayer,H<:Hamiltonian}
    term = gethamiltonian(hamiltonian(graph(layer)), H)
    local_idxs = idxs isa Colon ? graphidxs(layer) : idxs
    return setphysical!(layer, term, param, value; idxs = local_idxs)
end

function setphysical!(model, term::Hamiltonian, param::Symbol, value; idxs = Colon())
    units = _physical_parameter_units(term, param)
    converted = internalvalue(value, units, physicalscales(model), model; parameter = param)
    storage = getproperty(term, param)
    _assign_physical_value!(storage, converted, idxs)
    return storage
end

"""
    physicalvalue(model, term_type, param)

Return the current instantiated parameter value multiplied by its physical
scale metadata. Parameters without physical metadata are returned unchanged.
"""
function physicalvalue(model::M, ::Type{H}, param::Symbol) where {M<:AbstractIsingGraph,H<:Hamiltonian}
    term = gethamiltonian(hamiltonian(model), H)
    return physicalvalue(model, term, param)
end

function physicalvalue(layer::L, ::Type{H}, param::Symbol) where {L<:AbstractIsingLayer,H<:Hamiltonian}
    term = gethamiltonian(hamiltonian(graph(layer)), H)
    return physicalvalue(layer, term, param)
end

function physicalvalue(model, term::Hamiltonian, param::Symbol)
    units = _physical_parameter_units(term, param)
    storage = getproperty(term, param)
    return physicalvalue(storage, units, physicalscales(model), model; parameter = param)
end

export setphysical!, physicalvalue

"""
Linear annealing of a graph
"""
function anneal(g, total_time, Trange, steps)
    prev_time = time()
    time_per_step = total_time/steps
    @async for T in LinRange(Trange[1], Trange[2], steps)
        temp!(g, T)
        async_sleepy(time_per_step, prev_time)
        prev_time = time()
    end 
end
