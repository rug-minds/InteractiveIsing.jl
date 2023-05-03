
mutable struct LayerDefects
    layer::IsingLayer
    # graphdefects::GraphDefects
    ndefects::Int32
end

#extend base show for Layerdefects, showing only the number of defects
function Base.show(io::IO, defects::LayerDefects)
    print(io, "LayerDefects with $(ndefects(defects)) defects")
end

@setterGetter LayerDefects

maxdefects(defects::LayerDefects) = nStates(defects.layer)

graphdefects(df::LayerDefects) = defects(graph(layer(df)))

reset!(defects::LayerDefects) = ndefects(defects,0)

function getindex(defects::LayerDefects, idx)
    graphdefects(defects)[idxLToG(defects.layer, idx)]
end

function setindex!(defects::LayerDefects, val, idx)
    oldval = defects[idx]
    graphdefects(defects)[idxLToG(defects.layer, idx)] = val
    if val == true && oldval == false
        defects.ndefects += 1
    end
    return val
end

function setrange!(defects::LayerDefects, val, idxs)
    num_set = setrange!(graphdefects(defects), val, idxLToG.(Ref(layer(defects)), idxs))
    defects.ndefects += num_set
    return num_set
end



#NOT WORKING
function defectList(defects::LayerDefects)
    currentlayer = layer(defects)
    lidx = layeridx(currentlayer)
    g = graph(currentlayer)

    preceding_defects = precedingDefects(defects)

    return idxGToL.(Ref(currentlayer), defectList(graphdefects(defects))[(preceding_defects+1):(preceding_defects+ndefects(defects))])
end

function aliveList(defects::LayerDefects)
    currentlayer = layer(defects)
    lidx = layeridx(currentlayer)
    g = graph(currentlayer)

    preceding_states = precedingAlives(defects)
    
    return idxGToL.(Ref(currentlayer), aliveList(graphdefects(defects))[(preceding_states+1):(preceding_states + nStates(currentlayer) - ndefects(defects))] )
end

function precedingDefects(defects::LayerDefects)
    currentlayer = layer(defects)

    preceding_defects = 0
    for i in 1:(layeridx(currentlayer)-1)
        otherlayer = layerdefects(graphdefects(defects))[i]
        preceding_defects += ndefects(otherlayer)
    end
    return preceding_defects
end
export precedingDefects

function precedingAlives(defects::LayerDefects)
    currentlayer = layer(defects)

    preceding_states = 0
    for i in 1:(layeridx(currentlayer)-1)
        otherlayer = layerdefects(graphdefects(defects))[i]
        preceding_states += (nStates(layer(otherlayer)) - ndefects(otherlayer))
    end
    return preceding_states
end
export precedingAlives


