function init_connections_from_layers(precision, num_nodes, layers::AbstractLayerData...)
    cols = Int[]
    rows = Int[]
    vals = precision[]
    for layer in layers
        layercols, layerrows, layervals = gen_connections(layer, precision, get_weightgenerator(layer), num_nodes)
        append!(cols, layercols)
        append!(rows, layerrows)
        append!(vals, layervals)
    end
    return sparse(rows, cols, vals, num_nodes, num_nodes)
end
