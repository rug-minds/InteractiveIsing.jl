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

    sim = simulation[]
    newR = round(min(size(currentLayer(sim))...) / 10)

    setCircR!(sim, newR)
    layerName(sim)[] = name(currentLayer(sim))

    g = gs(sim)[1]

    delete!(mp["axis"], mp["image"])
    mp["sv_img_ob"][] = getSingleViewImg(g, ml)
    img_ob = mp["sv_img_ob"]
    mp["image"] = image!(mp["axis"], img_ob, colormap = :thermal, fxaa = false, interpolate = false)
    mp["image"].colorrange = stateset(currentLayer(sim))

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