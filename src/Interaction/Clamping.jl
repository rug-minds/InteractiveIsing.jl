function setClampIdxs(sim, layer, idxs, strenghts, cfac = 1)
    g = graph(sim, layer)
    clamps(g)[idxs] .= strenghts
    clampparam(g, cfac)

    setSimHType!(sim, layer, :Clamp => true)
end
export setClampIdxs


setClampFunc!(sim, layer, func, cfac = 1) = let vecs = functionToVecs(func, graph(sim, layer)); setClampIdxs(sim, layer, vecs[1], vecs[2], cfac) end
export setClampFunc!

function functionToVecs(func, g)
    m_matr = [Float32(func(;x,y)) for x in 1:g.N, y in 1:g.N]
    ([1:g.size;],reshape(transpose(m_matr),g.size))
end
    
function vecToImage(vec, length, width)
    imagesc(permutedims(reshape(vec, length, width)))
end
export vecToImage