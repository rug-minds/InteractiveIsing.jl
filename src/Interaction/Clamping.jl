function setClampIdxs(sim, idxs, strenghts, cfac = 1, g = currentLayer(sim))

    clamps(g)[idxs] .= strenghts
    clampparam(g, cfac)

    setSimHType!(sim, g, :Clamp => true)
end
export setClampIdxs


setClampFunc!(sim, layer, func, cfac = 1) = let vecs = functionToVecs(func, graph(sim, layer)); setClampIdxs(sim, layer, vecs[1], vecs[2], cfac) end
export setClampFunc!

function functionToVecs(func, g)
    m_matr = [Float32(func(;x,y)) for y in 1:glength(g), x in 1:gwidth(layer)]
    ([1:nStates(g);],reshape(m_matr,nStates(g)))
end
    
function vecToImage(vec, length, width)
    imagesc(reshape(vec, length, width))
end
export vecToImage