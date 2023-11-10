## Move this all to graphdefects, 
## just keep track of the number of defecs in a vector

struct LayerDefects
    g::IsingGraph
    l::IsingLayer
    layeridx::Integer
end

defects(layer::IsingLayer) = LayerDefects(graph(layer), layer, internal_idx(layer))
export defects
#extend base show for Layerdefects, showing only the number of defects
function Base.show(io::IO, defects::LayerDefects)
    print(io, "LayerDefects with $(ndefect(defects)) defects")
end

ndefect(defects::LayerDefects) = ndefect(defects.l)
@setterGetter LayerDefects

maxdefects(defects::LayerDefects) = nStates(defects.layer)

graphdefects(df::LayerDefects) = defects(df.g)

reset!(defects::LayerDefects) = ndefect(defects,0)

function getindex(d::LayerDefects, idx)
    defects(d.g)[idxLToG(idx, d.l)]
end

function setindex!(d::LayerDefects, val, idx)
    graphdefects(d)[idxLToG(idx, d.l)] = val
    return val
end




#NOT WORKING
function defectList(defects::LayerDefects)
    currentlayer = layer(defects)
    lidx = internal_idx(currentlayer)
    g = graph(currentlayer)

    preceding_defects = precedingDefects(defects)

    return idxGToL.(defectList(graphdefects(defects))[(preceding_defects+1):(preceding_defects+ndefect(defects))], Ref(currentlayer))
end

function aliveList(defects::LayerDefects)
    currentlayer = layer(defects)
    lidx = internal_idx(currentlayer)
    g = graph(currentlayer)

    preceding_states = precedingAlives(defects)
    
    return idxGToL.(aliveList(graphdefects(defects))[(preceding_states+1):(preceding_states + nStates(currentlayer) - ndefect(defects))], Ref(currentlayer) )
end

function precedingDefects(defects::LayerDefects)
    currentlayer = layer(defects)

    preceding_defects = 0
    for i in 1:(internal_idx(currentlayer)-1)
        otherlayer = layerdefects(graphdefects(defects))[i]
        preceding_defects += ndefect(otherlayer)
    end
    return preceding_defects
end
export precedingDefects

function precedingAlives(defects::LayerDefects)
    currentlayer = layer(defects)

    preceding_states = 0
    for i in 1:(internal_idx(currentlayer)-1)
        otherlayer = layerdefects(graphdefects(defects))[i]
        preceding_states += (nStates(layer(otherlayer)) - ndefect(otherlayer))
    end
    return preceding_states
end
export precedingAlives


