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
    dict["defects"] = defects(layer)
    dict["timers"] = timers(layer)
    dict["top"] = top(layer)

    dict["defects"].layer = nothing
    dict["top"].layer = nothing

    return LayerSavaData{layer_version}(dict)
end

Base.getindex(lsd::LayerSavaData, key::String) = lsd.data[key]
Base.keys(lsd::LayerSavaData) = keys(lsd.data)
Base.values(lsd::LayerSavaData) = values(lsd.data)
Base.length(lsd::LayerSavaData) = length(lsd.data)
Base.haskey(lsd::LayerSavaData, key::String) = haskey(lsd.data, key)


function IsingLayer(g, lsd::LayerSavaData)
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
    # l.defects = lsd.defects
    # l.timers = lsd.timers
    # l.top = lsd.top
    # l.defects.layer = l

    return l
end

# struct LayerSaveData_v1
#     T::Type
#     name::String
#     internal_idx::Int32
#     start::Int32
#     size::Tuple{Int32,Int32}
#     nstates::Int32
#     coords::Coords{Tuple{Int32,Int32,Int32}}
#     connections::Dict{Pair{Int32,Int32}, WeightGenerator}
#     defects::LayerDefects
#     timers::Vector{Timer}
#     top::LayerTopology
# end

# LayerSaveData_v1(layer::IsingLayer{T, GT}) where {T,GT} = LayerSaveData_v1(
#     T,
#     layer.name,
#     layer.internal_idx,
#     layer.start,
#     layer.size,
#     layer.nstates,
#     layer.coords,
#     layer.connections,
#     layer.defects,
#     layer.timers,
#     layer.top
# )

struct GraphSaveData_v1
    state::Union{Nothing, Vector{Float32}}
    sp_adj::SparseMatrixCSC{Float32,Int32}
    stype::SType
    layers::ShuffleVec{LayerSavaData}
    continuous::StateType
    defects::GraphDefects
    gdata::GraphData
end



function Base.show(io::IO, gsd::GraphSaveData_v1)
    print(io, "GraphSaveData")
end



function GraphSaveData_v1(g::IsingGraph)
    savelayers = ShuffleVec{LayerSavaData}()

    for layer in layers(g)
        push!(savelayers, LayerSavaData(layer))
    end

    savelayers.idxs .= layers(g).idxs

    gsd = GraphSaveData_v1(deepcopy(state(g)), deepcopy(sp_adj(g)), deepcopy(stype(g)), savelayers, deepcopy(continuous(g)), deepcopy(defects(g)), deepcopy(d(g)))

    gsd.defects.g = nothing
  
    return gsd
end

function IsingGraph(gsd::GraphSaveData_v1) 
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
    for layer_save in gsd.layers
        push!(g.layers, IsingLayer(g, layer_save))
    end
    g.layers.idxs .= gsd.layers.idxs

    # Reconstruct all pointers
    g.defects.g = g

    for layer in g.layers
        layer.graph = g
        layer.defects.layer = layer
    end

    return g
end
export GraphSaveData_v1

function saveGraph(g::IsingGraph)
    folder = dataFolderNow("Graphs")
    savedata = GraphSaveData_v1(g)
    filename = folder * "/g.jld2"
    save(filename, "g", savedata)
    @set_preferences!("last_save" => filename)
    return filename
end

loadGraph() = loadGraph(@load_preference("last_save"))
loadGraph(folder) = IsingGraph(load(folder)["g"])

export saveGraph, loadGraph