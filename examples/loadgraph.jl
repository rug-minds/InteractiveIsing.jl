using InteractiveIsing
using GLMakie

StateObs = Observable{Base.ReshapedArray{Float32, 2, SubArray{Float32, 1, Vector{Float32}, Tuple{UnitRange{Int32}}, true}, Tuple{}}}

const simulation = IsingSim(loadGraph())

const g = simulation(false);

createProcess(g)

const img_ob1 = Observable(state(g[1]))
const img_ob2 = Observable(state(g[2]))

obs = [img_ob1, img_ob2]

f = Figure()

function createSingleLayerView!(g, f)
    # Delete the previous axis
    try
        delete!(contents(f[1,2])[])
    catch
    end

    _grid = GridLayout(f[1,2], tellheight = false)
    _grid[1,1] = selector_buttons = GridLayout(_grid[1,1], tellheight = false)
    selected_layer_label = lift((x,y) -> "$x/$y", layerIdx(simulation), nlayers(simulation))
    selector_buttons[1,2] = Label(f, selected_layer_label, fontsize = 18)
    selector_buttons[1,1] = leftbutton = Button(f, label = "<", padding = (0,0,0,0), fontsize = 32, width = 60, height = 80)
    selector_buttons[1,3] = rightbutton = Button(f, label = ">", padding = (0,0,0,0), fontsize = 32, width = 60, height = 80)
    rowsize!(_grid[1,1].layout, 1, 80)
    img_ob = Observable{Base.ReshapedArray{Float32, 2, SubArray{Float32, 1, Vector{Float32}, Tuple{UnitRange{Int32}}, true}, Tuple{}}}(state(currentLayer(simulation)))

    on(leftbutton.clicks) do _
        changeLayer(-1)
        img_ob[] = state(currentLayer(simulation))
    end
    on(rightbutton.clicks) do _
        changeLayer(1)
        img_ob[] = state(currentLayer(simulation))
    end

    ax = Axis(_grid[2,:], xrectzoom = false, yrectzoom = false)
    image!(ax, img_ob, colormap = :thermal, fxaa = false, interpolate = false)
    return ax
end

createSingleLayerView!(g,f)

function createMultipleLayerView(g,f)



end
ax = Axis(f[1,2], xrectzoom = false, yrectzoom = false)
colsize!(f.layout, 1, 200)
rowsize!(f.layout, 1, Auto(400))

# Set images in the axis
im1 = image!(ax, img_ob1, colormap = :thermal, fxaa = false, interpolate = false)
im2 = image!(ax, img_ob2, colormap = :thermal, fxaa = false, interpolate = false)

# Toggle images on or off
im1.visible[] = true
im2.visible[] = true

# Translate the second image to the right, plus 50 pixels padding
translate!(im2, 250, 50)

# Add a pausing button
buttontext = lift(x -> x ? "Paused" : "Running", isPaused(simulation))
f[0,:] = pausebutton = Button(f, padding = (0,0,0,0), fontsize = 32, width = 120, height = 80, label = buttontext)
on(pausebutton.clicks) do _
    togglePause(g)
end

# Brush buttons
f[1,1] = buttons = GridLayout(tellheight = false)
buttons[1:3,1] = bs = [Button(f, padding = (0,0,0,0), fontsize = 32, width = 60, height = 80, label = "$i") for i in 1:-1:-1]
for (idx,val) in enumerate(1:-1:-1)
    on(bs[idx].clicks) do _
        # println("clicked $val")
        brush(sim(g))[] = Float32(val)
    end
end
# Clamp toggle
buttons[4,1] = clamptoggle = Toggle(f, active = false)
buttons[5,1] = Label(f, "Clamping", fontsize = 18)
# Size Slider
buttons[1:end-1,2] = rslider = Slider(f, range = 1:100, value = 10, horizontal = false)
rslider_text = lift(x -> "r: $x", brushR(simulation))
# Slider label
buttons[end,2] = Label(f, rslider_text , fontsize = 18)

function rslider_func(x)
    println("rslider.value[] = $(rslider.value[])")
        brushR(simulation)[] = Int32(rslider.value[])
        println(events(ax.scene).mousebutton[])
        println(events(f))
        println(events(ax.scene))
        # circ(sim, getOrdCirc(brushR(simulation)[]))

end

on(rslider.value) do x
    rslider_func(x)
end

function changeLayer(inc)
    setLayerIdx!(simulation, layerIdx(simulation)[] + inc)
    newR = round(min(size(currentLayer(simulation))...) / 10)

    setCircR!(simulation, newR)
    layerName(simulation)[] = name(currentLayer(simulation))
end

function createSingleLayerView!(g, f)
    # Delete the previous axis
    try
        delete!(contents(f[1,2])[])
    catch
    end

    _grid = GridLayout(f[1,2], tellheight = false)
    _grid[1,1] = selector_buttons = GridLayout(_grid[1,1], tellheight = false)
    selected_layer_label = lift((x,y) -> "$x/$y", layerIdx(simulation), nlayers(simulation))
    selector_buttons[1,2] = Label(f, selected_layer_label, fontsize = 18)
    selector_buttons[1,1] = leftbutton = Button(f, label = "<", padding = (0,0,0,0), fontsize = 32, width = 60, height = 80)
    selector_buttons[1,3] = rightbutton = Button(f, label = ">", padding = (0,0,0,0), fontsize = 32, width = 60, height = 80)
    rowsize!(_grid[1,1].layout, 1, 80)
    img_ob = Observable{Base.ReshapedArray{Float32, 2, SubArray{Float32, 1, Vector{Float32}, Tuple{UnitRange{Int32}}, true}, Tuple{}}}(state(currentLayer(simulation)))

    on(leftbutton.clicks) do _
        changeLayer(-1, simulation)
        img_ob[] = state(currentLayer(simulation))
    end
    on(rightbutton.clicks) do _
        changeLayer(1, simulation)
        img_ob[] = state(currentLayer(simulation))
    end

    ax = Axis(_grid[2,:], xrectzoom = false, yrectzoom = false)
    image!(ax, img_ob, colormap = :thermal, fxaa = false, interpolate = false)
    return ax
end

ax = createSingleLayerView!(g,f)

# Drawing on the axis
function MDrawCircle(buttons)
    if ispressed(ax.scene, Mouse.left)
        pos = mouseposition(ax.scene)
        drawCircle(g[1], pos[1], pos[2], brush(sim(g))[]; clamp = clamptoggle.active[])
    end
    return
end

on(events(ax.scene).mousebutton) do buttons
    MDrawCircle(buttons)
    return
end

display(f)

# f[3,1] = gl =  GridLayout()
# ax2 = Axis(gl[1,1])
# delete!(gl)

# Turn of fxaa for postprocessor
tm = Timer((timer) -> notify.(obs) ,0., interval = 1/60)

