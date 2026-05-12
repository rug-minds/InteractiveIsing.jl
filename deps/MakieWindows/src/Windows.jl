function LayerAxis(gridposition, l; colormap = :thermal, kwargs...)
    ax = Axis(gridposition, aspect = 1; kwargs...)
    ax.yreversed = @load_preference("makie_y_flip", default = false)
    img_ob = Observable(state(l))
    im = image!(ax, img_ob, fxaa = false, interpolate = false; colormap)
    im.colorrange[] = stateset(l)
    return ax, im, img_ob
end

"""
Make a new window with a figure and a screen
return the figure, screen and an observable for the window open state
"""
function empty_window(;kwargs...)
    f = Figure(;kwargs...);
    newscreen = GLMakie.Screen()
    display(newscreen, f)
    return f, newscreen, window_open(f)
end
export empty_window

"""
Window with an axis in the middle
"""
function axis_window(;pausebutton = true, kwargs...)
    window_type, kwargs = popkwarg(kwargs, :window_type, :Any)
    w = new_window(;window_type, kwargs...)
    ax = w[:ax] = Axis(w.f[1,1]; kwargs...)

    # Create a pausebutton
    if pausebutton
        pausebutton!(w)
    end

    return w
end

function axis3d_window(;pausebutton = true, kwargs...)
    window_type, kwargs = popkwarg(kwargs, :window_type, :Any)
    w = new_window(;window_type, kwargs...)
    ax = w[:ax] = Axis3(w.f[1,1]; kwargs...)

    # Create a pausebutton
    if pausebutton
        pausebutton!(w)
    end

    return w
end

function pausebutton!(w, gridpos = (0,1); kwargs...)
    pausetext = lift((p) -> p ? "Paused" : "Running", w[:paused])
    w[:pausebutton] = Button(w.f[gridpos[1], gridpos[2]][1,1], label = pausetext, tellwidth = false, height = 28)
    
    on(w[:pausebutton].clicks) do _
        w[:paused][] = !w[:paused][]
        if w[:paused][]
            close.(w.timers)
        else
            start.(w.timers)
        end
    end
    return w
end

"""
Create a pauseable timer that notifies an observable
"""
notifytimer(o::Observable; fps = 60) = PTimer((timer) -> notify(o), 0., interval = 1/fps)

# lines_window(x::AbstractVector, y::AbstractVector; kwargs...) = lines_window(Observable(x), Observable(y); kwargs...)

export new_window, axis_window, lines_window

function register_window(w::AbstractWindow, u::UUID)
    windows[u] = w
    on(window_open(w)) do x
        if !x
            delete!(windows, u)
        end
    end
end


function create_window_timer(update, isopen; fps = 60)
    timer = PTimer((timer) -> (@inline update()), 0., interval = 1/fps)
    on(isopen) do x
        if !x
            close(timer)
        end
    end
    return timer
end
export create_window_timer

"""
Return the window open observable for a figure
"""
function window_open(f::Figure)
    return events(f).window_open
end
export window_open



