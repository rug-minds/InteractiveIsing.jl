function rslider_func(x, sim)
    println("rslider.value[] = $(rslider.value[])")
        brushR(sim)[] = Int32(rslider.value[])
        println(events(ax.scene).mousebutton[])
        println(events(f))
        println(events(ax.scene))
        # circ(sim, getOrdCirc(brushR(simulation)[]))
end

function changeLayer(inc, sim)
    setLayerIdx!(sim, layerIdx(sim)[] + inc)
    newR = round(min(size(currentLayer(sim))...) / 10)

    setCircR!(sim, newR)
    layerName(sim)[] = name(currentLayer(sim))
end

# Drawing on the axis
function MDrawCircle(ax, buttons, sim)
    if ispressed(ax.scene, Mouse.left)
        pos = mouseposition(ax.scene)
        drawCircle(currentLayer(sim), pos[1], pos[2], brush(sim)[]; clamp = midpanel(ml)["clamptoggle"].active[])
    end
    return
end

function MDrawCircle2(ax, sim)
    pos = mouseposition(ax.scene)
    @async drawCircle(currentLayer(sim), pos[1], pos[2], brush(sim)[]; clamp = midpanel(ml)["clamptoggle"].active[])
    return
end

function showFig(f)
    screen = display(f)
    # resize!(screen, 1200, 1500)
end