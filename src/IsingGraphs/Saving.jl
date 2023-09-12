### Saving graph structures ###
const layer_version = 1
const graph_version = 1

struct LayerSavaData{Version} <: AbstractDict{String, Any}
    data::Dict{String, Any}
end

function LayerSavaData(layer::IsingLayer{T,GT}) where {T,GT}
    dict = Dict{String, Any}()
    dict["T"] = T
    dict["name"] = name(layer)
    dict["internal_idx"] = internal_idx(layer)
    dict["start"] = start(layer)
    dict["size"] = size(layer)
    dict["nstates"] = nstates(layer)
    dict["coords"] = coords(layer)
    dict["connections"] = connections(layer)
    dict["defects"] = deepcopy(defects(layer))
    dict["timers"] = deepcopy(timers(layer))
    dict["top"] = deepcopy(top(layer))

    dict["defects"].layer = nothing
    dict["top"].layer = nothing

    return LayerSavaData{layer_version}(dict)
end

Base.getindex(lsd::LayerSavaData, key::String) = lsd.data[key]
Base.keys(lsd::LayerSavaData) = keys(lsd.data)
Base.values(lsd::LayerSavaData) = values(lsd.data)
Base.length(lsd::LayerSavaData) = length(lsd.data)
Base.haskey(lsd::LayerSavaData, key::String) = haskey(lsd.data, key)
Base.iterate(lsd::LayerSavaData, state = 1) = iterate(lsd.data, state)

function IsingLayer(g, ls::LayerSavaData)
    lsd = ls.data
    l = IsingLayer(
        lsd["T"],
        g,
        lsd["internal_idx"],
        lsd["start"],
        lsd["size"]...,
        coords = Coords(lsd["coords"]),
        connections = lsd["connections"],
        name = lsd["name"]
    )
    l.defects = lsd["defects"]
    l.timers = lsd["timers"]
    l.top = lsd["top"]
    l.defects.layer = l
    l.top.layer = l

    return l
end

struct GraphSaveData{Version}
    state::Union{Nothing, Vector{Float32}}
    sp_adj::SparseMatrixCSC{Float32,Int32}
    stype::SType
    shuffle_idxs::Vector{Int}
    continuous::StateType
    defects::GraphDefects
    gdata::GraphData
end



function Base.show(io::IO, gsd::GraphSaveData)
    print(io, "GraphSaveData")
end

function GraphSaveData(g::IsingGraph)

    gsd = GraphSaveData{graph_version}(deepcopy(state(g)), deepcopy(sp_adj(g)), deepcopy(stype(g)), layers(g).idxs, deepcopy(continuous(g)), deepcopy(defects(g)), deepcopy(d(g)))

    gsd.defects.g = nothing
  
    return gsd
end

function IsingGraph(sd::Dict{String, Any})
    gsd = sd["gsd"]
    # Initialize graph
    g = IsingGraph(
        gsd.state,
        gsd.sp_adj,
        gsd.stype,
        ShuffleVec{IsingLayer}(),
        gsd.continuous,
        gsd.defects,
        gsd.gdata
    )

    # Reconstruct layer from layer save data
    for layer in 1:sd["num_layers"]
        push!(g.layers, IsingLayer(g, sd["layer_$layer"]))
    end
    g.layers.idxs .= gsd.shuffle_idxs

    # Reconstruct all pointers
    g.defects.g = g

    return g
end
export GraphSaveData

function saveGraph(g::IsingGraph; savepref = true)
    folder = dataFolderNow("Graphs")
    savedata = Dict{String, Any}()
    savedata["gsd"] = GraphSaveData(g)
    savedata["num_layers"] = length(g.layers)
    for (idx, layer) in enumerate(g.layers.data)
        savedata["layer_$idx"] = LayerSavaData(layer)
    end
    filename = folder * "/g.jld2"
    save(filename, savedata)
    if savepref
        @set_preferences!("last_save" => filename)
    end
    return filename
end

loadGraph() = loadGraph(@load_preference("last_save"))
function loadGraph(folder)
    savedata = load(folder)
    IsingGraph(savedata)
end
export saveGraph, loadGraph