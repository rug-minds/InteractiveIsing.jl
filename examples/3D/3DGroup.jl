using InteractiveIsing

wginner = @WG "(dx,dy)->1/(((2dy)^2+(2dx)^2))" NN = 1
wgouter = @WG "(dx,dy,dz)->1/((dy^2+(2dx)^2+dz^2))" NN = 2
function rect3D(g::IsingGraph, length, width, height; type = Continuous, set = (-1,1), wginner = nothing, wgouter = nothing)
    for layer_idx in 1:height
        newlayer = addLayer!(g, length, width; type, set)
        setcoords!(newlayer, z = layer_idx)
       
    end
    for layer_idx in 1:height
        if !isnothing(wginner)
            println("Connecting idx $layer_idx with weight $wginner")
            genAdj!(g[layer_idx], wginner)
        end
        if !isnothing(wgouter) && layer_idx != 1
            println("Connecting idx $layer_idx with weight $wgouter")
            genAdj!(g[layer_idx-1], g[layer_idx], wgouter)
        end
    end
end

g = IsingGraph()

rect3D(g, 50, 50, 50, type = Discrete; wginner, wgouter)

simulate(g)
