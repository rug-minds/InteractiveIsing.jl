using InteractiveIsing

wginner = @WG "(dx,dy)->1/(((2dy)^2+(2dx)^2))" NN = 1
wgouter = @WG "(dx,dy,dz)->1/((dy^2+(2dx)^2+dz^2))" NN = 2
function cuboidLayer(g::IsingGraph, length, width, height; type = Continuous, set = (-1,1), wginner = nothing, wgouter = nothing)
    
    newlayer = addLayer!(g, length, width, height; type, set)
    # setcoords!(newlayer, z = 0)

    row_idxs = Int[]
    col_idxs = Int[]
    weights = Float64[]

    for layer_idx in eachindex(newlayer)
        if !isnothing(wginner)
            println("Connecting idx $layer_idx with weight $wginner")
            # genAdj!(g[layer_idx], wginner)
            new_r, new_c, new_w = InteractiveIsing.genLayerConnections(newlayer[layer_idx], wginner)
            append!(row_idxs, new_r)
            append!(col_idxs, new_c)
            append!(weights, new_w)
        end
        if !isnothing(wgouter) && layer_idx != 1
            println("Connecting idx $layer_idx with weight $wgouter")
            # genAdj!(g[layer_idx-1], g[layer_idx], wgouter)
            new_r, new_c, new_w = InteractiveIsing.genLayerConnections(newlayer[layer_idx-1], newlayer[layer_idx], wgouter)
            append!(row_idxs, new_r)
            append!(col_idxs, new_c)
            append!(weights, new_w)
        end
    end
    InteractiveIsing.set_adj!(g, (row_idxs, col_idxs, weights))
end

g = IsingGraph()

cuboidLayer(g, 40, 40, 60, type = Discrete; wginner, wgouter)

simulate(g)