export save_isinggraph, load_isinggraph

const ISINGGRAPH_JLD2_KEY = "isinggraph"

"""
    save_isinggraph(path, graph; key = "isinggraph") -> path

Save a complete `AbstractIsingGraph` object to a JLD2 file.

This stores the graph object itself, including state, adjacency, Hamiltonian,
temperature, index set, and layer metadata. The file can be restored with
[`load_isinggraph`](@ref) from the same package version.
"""
function save_isinggraph(path::AbstractString, graph::AbstractIsingGraph; key::AbstractString = ISINGGRAPH_JLD2_KEY)
    mkpath(dirname(path))
    JLD2.save(path, key, graph)
    return path
end

"""
    load_isinggraph(path; key = "isinggraph") -> AbstractIsingGraph

Load a graph previously written with [`save_isinggraph`](@ref).
"""
function load_isinggraph(path::AbstractString; key::AbstractString = ISINGGRAPH_JLD2_KEY)
    graph = JLD2.load(path, key)
    graph isa AbstractIsingGraph || error("JLD2 entry \"$key\" in $path is not an AbstractIsingGraph; got $(typeof(graph))")
    return graph
end

"""
    JLD2.save(path, graph::AbstractIsingGraph; key = "isinggraph") -> path

Convenience extension so a graph can be saved directly with `JLD2.save`.
"""
function JLD2.save(path::AbstractString, graph::AbstractIsingGraph; key::AbstractString = ISINGGRAPH_JLD2_KEY)
    return save_isinggraph(path, graph; key)
end
