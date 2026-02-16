
export overlayNoise!, resetstate!, activateone!
# Setting elements

"""
Set spins either to a value or clamp them
"""
#Clean this up
# TODO: Shouldn't always reprepare sim if the iterator didn't change
function setSpins!(g::AbstractIsingGraph{T}, idxs::Union{AbstractArray{<:Integer}, AbstractArray{<:Integer}, <:UnitRange, AbstractArray{<:CartesianIndex}}, val, clamp::Bool = false) where T
    hasdefects_before = hasDefects(graph(g))

    # Set the defects
    clamprange!(g, clamp, idxs)

    hasdefects_after = hasDefects(graph(g))
    if hasdefects_before != hasdefects_after
        reprepare(graph(g))
    end

    # Set the spins
    copystate(g, idxs, val)
end

function setdefect!(g::AbstractIsingLayer{T,D}, vals, idxs) where {T,D}

end

copystate(g::AbstractIsingGraph, idxs, val::Real) = @inbounds state(g)[idxs] .= closestTo(g, val)

copystate(g::AbstractIsingGraph, idxs, val::AbstractArray) = @inbounds state(g)[idxs] .= val[1:end]

copystate(g::IsingLayer, idxs, val::AbstractArray) = mapToStateSet!(g, (@view state(g)[idxs]), val)



setSpins!(g::AbstractIsingGraph, vals::AbstractArray, clamp::Bool = false) = setSpins!(g, graphidxs(g), vals, clamp)

function setSpins!(g::AbstractIsingGraph, coords::Union{Vector{NTuple{3,Int}}, Vector{NTuple{2,Int}}}, val::Real, clamp::Bool = false)
    setSpins!(g, coordToIdx.(coords, Ref(size(g))), val, clamp)
end

function setSpins!(g::AbstractIsingGraph{T}, idx::Integer, val::Real, clamp::Bool = false) where T
    hasdefects_before = hasDefects(graph(g))
    
    setdefect(g, clamp, idx)

    hasdefects_after = hasDefects(graph(g))

    if hasdefects_before != hasdefects_after
        reprepare(graph(g))
    end

    @inbounds state(g)[idx] = val
end

function setDefects!(g, val, idxs)
    hasdefects_before = hasDefects(graph(g))
    
    defects(g)[idxs] = val
    
    hasdefects_after = hasDefects(graph(g))
    
    if hasdefects_before != hasdefects_after
        reprepare(graph(g))
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
        reprepare(g)
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
resetstate!(g::IsingGraph) = state(g) .= initRandomState(g)
resetstate!(l::IsingLayer) = state(l)[:] .= rand(l, length(state(l)))
#TODO: This is a shitty implementation
resetstate!(layers::IsingLayer...) = for l in layers; resetstate!(l); end

"""
For a layer, set all to zero and 1 to 1
"""
activateone!(l::IsingLayer, idx, val = 1, allval = 0) = begin state(l) .= allval; state(l)[idx] = val end
"""
Set the temperature and notify the simulation
"""
function settemp(g,val)
    temp(g, val)
    if !isnothing(sim(g))
        temp(sim(g), val)
    end
end
export settemp

"""
Linear annealing of a graph
"""
function anneal(g, total_time, Trange, steps)
    prev_time = time()
    time_per_step = total_time/steps
    @async for T in LinRange(Trange[1], Trange[2], steps)
        temp(g, T)
        async_sleepy(time_per_step, prev_time)
        prev_time = time()
    end 
end
