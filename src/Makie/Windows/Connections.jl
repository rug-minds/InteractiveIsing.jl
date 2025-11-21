export visualize_connections
function make_visualize_connections(g::AbstractIsingLayer, i::Integer; adj = InteractiveIsing.adj(g))
    window = new_window(window_type = :Connections, title = "Connections from node $i")
    # Make GridLayout
    window[:gridlayout] = window.f[1, 1] = GridLayout()

    colorvec = zeros(RGBAf, nstates(g))
    weights = conn_weights(adj,i)
    colorvec[conn_idxs(adj,i)] .= weight_colors(weights)
    # return (colorvec)
    window[:layer] = g
    window[:ax] = create_layer_axis!(window, window, color = colorvec)
end

current_layer(window::MakieWindow{:Connections}) = window[:layer]


# visualize_connections(g, i, j) = visualize_connections(g, coordToIdx(Int32.((i, j)), size(g)))
# visualize_connections(g, i, j, k) = visualize_connections(g, coordToIdx(Int32.((i, j, k))..., size(g)) )
visualize_connections(l, coords::Int...; adj = InteractiveIsing.adj(l)) = make_visualize_connections(l, LinearIndices(l)[coords...]; adj)


function weight_colors(weights)
    max_weight = maximum(weights)
    min_weight = minimum(weights)
    colorvec = zeros(RGBAf, length(weights))
    for w_idx in eachindex(weights)
        #linear interpolation from purple to yellow
        colorvec[w_idx] = RGBAf(0.5, 0.0, 0.5 + 0.5*(weights[w_idx] - min_weight)/(max_weight - min_weight), 0.1 + 0.9*(weights[w_idx] - min_weight)/(max_weight - min_weight))
    end
    return colorvec
end