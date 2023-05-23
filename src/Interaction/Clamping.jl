function setClampIdxs(g::IsingGraph, idxs, strenghts, cfac = 1)
    # println(strenghts)
    clamps(g)[idxs] .= strenghts
    clampparam(g, cfac)

    setSimHType!(sim(g), :Clamp => true)
end
export setClampIdxs


setClampFunc!(layer, func, cfac = 1) = let vecs = functionToVecs(func, layer); setClampIdxs(graph(layer), vecs..., cfac) end
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