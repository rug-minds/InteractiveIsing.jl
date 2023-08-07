function remClamp!(layer::IsingLayer)
    clamps(layer) .= 0

    # If there are no clamps, set Clamp to false
    isnothing(findfirst(x -> x != 0, clamps(graph(layer) ))) && setSType!(layer, :Clamp => false)
end

function remClamp!(g::IsingGraph)
    clamps(g) .= 0
    clampparam(g, 0)
    setSType!(g, :Clamp => false)
    return
end

function setClampIdxs!(g::IsingGraph, idxs, strenghts, cfac = 1)
    # println(strenghts)
    clamps(g)[idxs] .= strenghts
    clampparam(g, cfac)

    setSType!(g, :Clamp => true)
    return
end
export setClampIdxs!


function setClampFunc!(layer, func, cfac = 1)
    _clamps = clamps(layer)
    clampparam(graph(layer), cfac)

    for i in 1:size(clamps)[1]
        for j in 1:size(clamps)[2]
            _clamps[i,j] = func(;x=j,y=i)
        end
    end
    return
end
export setClampFunc!

function functionToVecs(func, g)
    m_matr = [Float32(func(;x,y)) for y in 1:glength(g), x in 1:gwidth(g)]
    ([1:nStates(g);], m_matr[:])
end
export functionToVecs
    
function vecToImage(vec, length, width)
    imagesc(reshape(vec, length, width))
end
export vecToImage