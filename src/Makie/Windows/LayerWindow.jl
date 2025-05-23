mutable struct LayerWindow <: AbstractWindow
    l::IsingLayer    
    f::Figure
    screen::GLMakie.Screen
    other::Dict{Symbol, Any}
    timer::PTimer
    LayerWindow(a,b,c,d) = new(a,b,c,d)
end

pushtype(::LayerWindow) = :multiple
closetimers(w::LayerWindow) = close(w.timer)


function LayerWindow(l::IsingLayer)
    ml = getml()
    f, screen, isopen = empty_window()
    grid = GridLayout(f[1,1])
    ax, im, img_ob = LayerAxis(grid[1,1], l, title = "Layer $(name(l))", tellwidth = false, tellheight = false)
    timer = create_window_timer(() -> notify(img_ob), isopen)

    lw = LayerWindow(l, f, screen, Dict{Symbol, Any}())
    push!(ml.windowlist, lw)
    lw.timer = timer

    push!(windowlist, lw)
    return lw    
end
export LayerWindow

