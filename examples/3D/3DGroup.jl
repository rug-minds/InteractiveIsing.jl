using InteractiveIsing

wginner = @WG "(dx,dy)->-1/(((2dy)^2+(2dx)^2))" NN = 1
wgouter = @WG "(dx,dy,dz)->1/((dy^2+(2dx)^2+dz^2))" NN = 1
function cuboidLayer(g::IsingGraph, length, width, height; type = Continuous, set = (-1,1), wginner = nothing, wgouter = nothing)
    
    newlayer = addLayer!(g, length, width, height; type, set)
    # setcoords!(newlayer, z = 0)

    T = eltype(g)

    row_idxs = Int32[]
    col_idxs = Int32[]
    weights = T[]

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

cuboidLayer(g, 20, 20, 40, type = Discrete)

# setParam!(g[1][1], :b, 1, true)

# setParam!(g[1], :b, 0, true)

function set3DAdj!(layer3d, wginner = nothing, wgouter = nothing)

    T = eltype(layer3d)
    row_idxs = Int32[]
    col_idxs = Int32[]
    weights = T[]

    for layer_idx in eachindex(layer3d)
        if !isnothing(wginner)
            println("Connecting idx $layer_idx with weight $wginner")
            # genAdj!(g[layer_idx], wginner)
            new_r, new_c, new_w = InteractiveIsing.genLayerConnections(layer3d[layer_idx], wginner)
            append!(row_idxs, new_r)
            append!(col_idxs, new_c)
            append!(weights, new_w)
        end
        if !isnothing(wgouter) && layer_idx != 1    
            println("Connecting idx $layer_idx with weight $wgouter")
            new_r, new_c, new_w = InteractiveIsing.genLayerConnections(layer3d[layer_idx-1], layer3d[layer_idx], wgouter)
            append!(row_idxs, new_r)
            append!(col_idxs, new_c)
            append!(weights, new_w)
        end
    end
    InteractiveIsing.set_adj!(g, (row_idxs, col_idxs, weights))
end

simulate(g)

# wginner = @WG "(dx,dy)->-2/(((2dy)^2+(2dx)^2))" NN = 1
# # wgouter = @WG "(dx,dy,dz)->1/((dy^2+(2dx)^2+dz^2))" NN = 1
# wgouter = @WG "(dx,dy,dz) -> (dx != 0 || dy != 0) ? 0 : 1/(dz^2)" NN = 3

wg = @WG "(dx,dy,dz) -> 1/(dx^2+dy^2+dz^2)" NN = (2,2,5)
genAdj!(g[1], wg)

# set3DAdj!(g[1], wginner, wgouter)

setParam!(g[1], :b, 2, true)
# setParam!(g[1][1], :b , 3, true)
# setParam!(g[1][60], :b , 3, true)

function TrianglePulseB(g, t, amp = 1, steps = 1000)
    first = LinRange(0, amp, floor(Int,steps/4))
    second = LinRange(amp, 0, floor(Int,steps/4))
    third = LinRange(0, -amp, floor(Int,steps/4))
    fourth = LinRange(-amp, 0, floor(Int,steps/4))
    pulse = vcat(first, second, third, fourth)

    tstep = t/steps

    t_i = time()
    for idx in 1:steps
        while time() - t_i < tstep
            
        end
        setParam!(g, :b, pulse[idx], true)
        t_i = time()
    end

end

# Threads.@spawn TrianglePulseB(g, 10, 5)