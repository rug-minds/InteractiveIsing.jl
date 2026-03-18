export addRandomDefects!, setClamp!, remClamp!

"""
Randomly make p percent of states in layer defect
"""
function addRandomDefects!(layer, p, val = 0)
    println("Adding random defects to layer")

    nonzeros = Int32[]
    for i in 1:length(state(layer))
        if nand(state(layer)[i] == 0, isDefect(graph(layer))[idxLToG(i, layer)] )
            push!(nonzeros, i)
        end
    end

    # If is empty do nothing
    isempty(nonzeros) && return nothing

    # From nonzeros or non-defects choose p% to set to 0 and defect
    idxs = nonzeros[sample([true, false], Weights([p/100,1-p/100]), length(nonzeros))]

    setSpins!(layer, idxs, val, true)

    reinit(graph(layer))

    return
end

"""
Add a term to the Hamiltonian of the form beta 1/2(sigma_i - y)^2 where sigma_i is the i-th state and y is the target for that state
Give a function in the form of (x,y)
"""
setClamp!(layer::AbstractIsingLayer, func::Function) = setClampFunc!(layer, (;x,y) -> func(x,y))
setClamp!(layer::AbstractIsingLayer, idxs::Vector, strengths::Vector) = setClampIdxs!(layer, idxs, strengths)
function setClamp!(l::AbstractIsingLayer, idx::Integer, strength::Real)
    defects(l)[idx] = true
    state(l)[idx] = 1
    return state(l)
end
function setClamp!(l::AbstractIsingLayer, strength::Real)
    setdefectrange!(l, Int32[1:length(state(l));])
    state(l) .= strength
    return state(l)
end
#remClamp!

function globalB!(g, strength)
    bfield(g) .= strength
end

function setB!(g::IsingGraph, val::Real, idxs::AbstractVector)
    bfield(g)[idxs] .= val
end

setB!(l::IsingLayer, val) = setB!(graph(l), val, graphidxs(l))
setB!(val::Real, ls::IsingLayer...) = for l in ls; setB!(l, val); end
setB!(l::IsingLayer, strengths::AbstractVector) = setBIdxs!(graph(l), (@view graphidxs(l)[1:length(strengths)]), strengths)
export globalB!, setB!

avgB(l::IsingLayer) = sum(bfield(l))/length(bfield(l))
export avgB