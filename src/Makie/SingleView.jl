export singleView
max_r_slider(simulation, slider) = slider.range[] = 1:(floor(Int64,min(size(currentLayer(simulation))...)/2))
sim_max_r(simulation) = (floor(Int64,min(size(currentLayer(simulation))...)/2))
function getSingleViewImg(g, ml)
    simulation = sim(g)
    if midpanel(ml)["showbfield"].active[]
        return bfield(currentLayer(simulation))
    else
        return state(currentLayer(simulation))
    end
end
function liftcouple(f, obs::AbstractObservable, args...)
    newob = lift(f, obs, args...)
    append!(newob.inputs, [obs, args...])
    return newob
end
function findcallback(input, obs)
    for (l_idx, listener_pair) in enumerate(input.listeners)
        listener = listener_pair[2]
        if typeof(listener) == Observables.MapCallback
            if listener.result == obs
                return l_idx
            end
        end
    end
    return nothing
end

function decouple!(obs::AbstractObservable)
    for (i_idx, input_ob) in enumerate(obs.inputs)
        if typeof(input_ob) <: Observable
            callback_idx = findcallback(input_ob, obs)
            deleteat!(input_ob.listeners, callback_idx)
            deleteat!(obs.inputs, i_idx)
            return true
        end
    end
    return false
end
"""
"""
function singleView(ml, g)
    mp = midpanel(ml)
    f = fig(ml)
    simulation = sim(g)

    ml["current_view"] = singleView

    obs_funcs = etc(ml)["obs_funcs"] = []
    coupled_obs = etc(ml)["coupled_obs"] = Observable[]

    # LAYER SELECTOR  BUTTONS
    toppanel(ml)["sb"] = selector_buttons = GridLayout(toppanel(ml)["mid_grid"][3,1], tellwidth = false)
    selected_layer_label = liftcouple((x,y) -> "$x/$y", layerIdx(simulation), nlayers(simulation))
    push!(coupled_obs, selected_layer_label)

    toppanel(ml)["sb_<"] = selector_buttons[1,1] = leftbutton = Button(f, label = "<", padding = (0,0,0,0), fontsize = 14, width = 40, height = 28)
    toppanel(ml)["sb_label"] = selector_buttons[1,2] = Label(f, selected_layer_label, fontsize = 18)
    toppanel(ml)["sb_>"] = selector_buttons[1,3] = rightbutton = Button(f, label = ">", padding = (0,0,0,0), fontsize = 14, width = 40, height = 28)
    # rowsize!(_grid[1,1].layout, 1, 80)

    # ISING IMAGE
    mp["sv_img_ob"] = img_ob = Observable{Base.ReshapedArray}(getSingleViewImg(g, ml))
    push!(obs_funcs, on(midpanel(ml)["showbfield"].active, weak = true) do _
        img_ob[] = getSingleViewImg(g, ml)
    end)
    # img_ob = Observable{Base.ReshapedArray{Float32, 2, SubArray{Float32, 1, Vector{Float32}, Tuple{UnitRange{Int32}}, true}, Tuple{}}}(state(currentLayer(simulation)))

    # max_r_slider(simulation, midpanel(ml)["rslider"]) 

    push!(obs_funcs, on(leftbutton.clicks, weak = true) do _
        changeLayer(-1, simulation)
        img_ob[] = getSingleViewImg(g, ml)
        # max_r_slider(simulation, midpanel(ml)["rslider"]) 
        reset_limits!(ax)
    end)

    push!(obs_funcs, on(rightbutton.clicks, weak = true) do _
        changeLayer(1, simulation)
        img_ob[] = getSingleViewImg(g, ml)
        # max_r_slider(simulation, midpanel(ml)["rslider"]) 
        reset_limits!(ax)
    end)
    # ax = Axis(mp[][1,2], xrectzoom = false, yrectzoom = false, aspect = DataAspect(), ypanlock = true, xpanlock = true, yzoomlock = true, xzoomlock = true)
    ax = Axis(mp[][1,2], xrectzoom = false, yrectzoom = false, aspect = DataAspect(), tellheight = true)
    im = image!(ax, img_ob, colormap = :thermal, fxaa = false, interpolate = false)

    # rowsize!(_grid, 1, Relative(1/20))

    mp["axis"] = ax
    mp["image"] = im
    mp["obs"] = img_ob

    push!(obs_funcs, on(events(ax.scene).mousebutton, weak = true) do buttons
        MDrawCircle(ax, buttons, simulation)
        return
    end)
    # me_axis = addmouseevents!(ax.scene)

    # onmouseleftdown(me_axis) do ev
    #     println("Dragging stopped")
    #     mp["me_axis"] = me_axis
    #     return
    # end
    # onmouseleftdragstop(me_axis) do _
    #     println("Dragging stopped")
    #     # close(mp["mousedrag_timer"])
    #     return
    # end

    # Weightgenerator display
    wg_label_obs = liftcouple(x-> "$(wg(g[x]))", layerIdx(simulation))
    push!(coupled_obs, wg_label_obs)
    bp = bottompanel(ml)
    bp_midgrid_toprow = 1+bp["mid_grid"].offsets[1]
    bottompanel(ml)["wf_label"] = Label(bp["mid_grid"][bp_midgrid_toprow - 1,1], wg_label_obs, fontsize = 12)

    if haskey(etc(ml), "timedfunctions_timer")
        close(etc(ml)["timedfunctions_timer"])
    end

    etc(ml)["timedfunctions_timer"] = Timer((timer) -> (notify(mp["obs"]); timedFunctions(simulation)) ,0., interval = 1/60)

    return
end
create_singleview = singleView

using GLMakie.Makie.GridLayoutBase: deleterow!
function cleanup(ml, ::typeof(singleView))
    # Observables
    etc(ml)["obs_funcs"] = []
    for obs in etc(ml)["coupled_obs"]
        println(decouple!(obs))
    end
    etc(ml)["coupled_obs"] = Observable[]


    # Selector buttons
    sb = toppanel(ml)["sb"]
    for idx in 1:length(sb.content)
        c = first(sb.content).content
        delete!(c)
    end

    tp = toppanel(ml)
    deleterow!(tp["sb"].parent, 3)
    delete!(tp, "sb", "sb_<", "sb_label", "sb_>")

    # Axis
    mp = midpanel(ml)
    delete!(mp["axis"])
    delete!(mp, "axis", "image")

    # Weightgenerator display
    bp = bottompanel(ml)
    delete!(bp["wf_label"])
    delete!(bp, "wf_label")
    bp_midgrid_toprow = 1+bp["mid_grid"].offsets[1] 
    deleterow!(bp["mid_grid"], bp_midgrid_toprow)

    #Timer
    close(etc(ml)["timedfunctions_timer"])
    delete!(etc(ml), "timedfunctions_timer")

    ml["current_view"] = nothing
    

end
export cleanup
