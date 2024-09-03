function rslider_func(x, sim)
    println("rslider.value[] = $(rslider.value[])")
        brushR(sim)[] = Int32(rslider.value[])
        println(events(ax.scene).mousebutton[])
        println(events(f))
        println(events(ax.scene))
        # circ(sim, getOrdCirc(brushR(simulation)[]))
end

function setLayerSV(idx)
    ml = getml()
    mp = midpanel(ml)

    # dim = glength(currentLayer(simulation[]))

    sim = simulation[]
    cur_layer = currentLayer(sim)

    newR = round(min(size(currentLayer(sim))...) / 10)

    setCircR!(sim, newR)
    layerName(sim)[] = name(cur_layer)

    g = gs(sim)[1]

    delete!(mp["axis"], mp["image"])
    mp["sv_img_ob"][] = img_ob = getSingleViewImg(g, ml)
    # img_ob = mp["sv_img_ob"]

    cur_layer = cur_layer
    # create_layer_axis(cur_layer, mp)
    new_img!(g, cur_layer, mp)
    mp["image"].colorrange = stateset(cur_layer)

    reset_limits!(mp["axis"])
end

# Drawing on the axis
function MDrawCircle(ax, buttons, sim)
    ml = getml()
    if ispressed(ax.scene, Mouse.left)
        pos = mouseposition(ax.scene)
        drawCircle(currentLayer(sim), pos[1], pos[2], brush(sim)[]; clamp = midpanel(ml)["clamptoggle"].active[])
    end
    return
end

function MDrawCircle2(ax, sim)
    ml = getml()
    pos = mouseposition(ax.scene)
    @async drawCircle(currentLayer(sim), pos[1], pos[2], brush(sim)[]; clamp = midpanel(ml)["clamptoggle"].active[])
    return
end

function showFig(f)
    screen = display(f)
    # resize!(screen, 1200, 1500)
end

""" 
Function to create an unsafe vector from the view
"""
function create_unsafe_vector(view_array)
    # Get the pointer to the view array
    ptr = pointer(view_array)
    # Wrap the pointer into a Julia array without copying
    unsafe_vector = unsafe_wrap(Vector{eltype(view_array)}, ptr, length(view_array))
    return unsafe_vector
end


# Maybe I can just make makie get the latest image always?
function getSingleViewImg(g, ml)
    simulation = sim(g)
    mp = midpanel(ml)
    if midpanel(ml)["showbfield"].active[]
        return bfield(currentLayer(simulation))
    else
        return state(currentLayer(simulation))
    end
end

function flip_y_axis()
    @set_preferences!("makie_y_flip" => !(@load_preference("makie_y_flip", default = false)))
    try
        ml = getml()
        midpanel(ml)["axis"].yreversed[] = @load_preference("makie_y_flip", default = false)
    catch
    end
end

getgrid(lp::LayoutPanel) = lp.panel
getgrid(window::MakieWindow) = window["gridlayout"]

function create_layer_axis!(layer, panel_or_window; kwargs...)
    layerdim = length(size((layer)))
    g = graph(layer)
    grid = getgrid(panel_or_window)
    if layerdim == 2
        panel_or_window["axis"] = ax = Axis(grid[1,2], xrectzoom = false, yrectzoom = false, aspect = DataAspect(), tellheight = true)
        # TODO: Set colorrange based on the type of layer
    else
        panel_or_window["axis"] = ax = Axis3(grid[1,2], tellheight = true)
        # TODO: 3D BField
    end
    panel_or_window["axis"].yreversed = @load_preference("makie_y_flip", default = false)

    new_img!(g, layer, panel_or_window)
end

# Get image either for 2d or 3d
function new_img!(g, layer, mp)
    dims = length(size(layer))
    ax = mp["axis"]

    if dims == 2
        mp["obs"] = Observable(getSingleViewImg(g, getml()))
        mp["image"] = image!(ax, mp["obs"], colormap = :thermal, fxaa = false, interpolate = false)
    elseif dims == 3
        unsafe_view = create_unsafe_vector(@view state(g)[graphidxs(layer)])
        sz = size(layer)
        allidxs = [1:length(state(layer));]
        xs = idx2xcoord.(Ref(sz), allidxs)
        ys = idx2ycoord.(Ref(sz), allidxs)
        zs = idx2zcoord.(Ref(sz), allidxs)
        obs = mp["obs"] = Observable(unsafe_view)
        mp["image"] = meshscatter!(ax, xs, ys, zs, markersize = 0.2, color = obs, colormap = :thermal)
    end

    mp["image"].colorrange[] = stateset(layer)
end

# function create_layer_axis!(layer, panel_or_window; color = nothing, pos = (1,1), colormap = :thermal)
#     layerdim = length(size((layer)))
#     g = graph(layer)
#     grid = getgrid(panel_or_window)
#     if layerdim == 2
#         panel_or_window["axis"] = ax = Axis(grid[pos[1],pos[2]], xrectzoom = false, yrectzoom = false, aspect = DataAspect(), tellheight = true)
#         # TODO: Set colorrange based on the type of layer
#     else
#         panel_or_window["axis"] = ax = Axis3(grid[pos[1],pos[2]], tellheight = true)
#         # TODO: 3D BField
#     end
#     panel_or_window["axis"].yreversed = @load_preference("makie_y_flip", default = false)

#     new_img!(g, layer, panel_or_window; color, colormap)
# end

# # Get image either for 2d or 3d
# function new_img!(g, layer, mp; color = nothing, colormap = :thermal)
#     dims = length(size(layer))
#     ax = mp["axis"]

#     if dims == 2
#         color = isnothing(color) ? (mp["obs"] = Observable(getSingleViewImg(g, getml()))) : color
#         mp["image"] = image!(ax, color, colormap = colormap, fxaa = false, interpolate = false)
#     elseif dims == 3
#         if isnothing(color)
#             color = Observable(getSingleViewImg(g, getml()))
#             mp["obs"] = color
#         else
#             unsafe_view = create_unsafe_vector(@view state(g)[graphidxs(layer)])
#             mp["obs"] = Observable(unsafe_view)
#         end

       
#         sz = size(layer)
#         allidxs = [1:length(state(layer));]
#         xs = idx2xcoord.(Ref(sz), allidxs)
#         ys = idx2ycoord.(Ref(sz), allidxs)
#         zs = idx2zcoord.(Ref(sz), allidxs)
        
#         println("Color", color)
#         mp["image"] = meshscatter!(ax, xs, ys, zs, markersize = 0.2, color = mp["obs"], colormap = colormap)
#     end

#     mp["image"].colorrange[] = stateset(layer)
# end