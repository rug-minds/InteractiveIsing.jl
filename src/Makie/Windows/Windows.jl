abstract type AbstractWindow end
# Is this type of window unique?
pushtype(::W) where W<:AbstractWindow = pushtype(W) 
function closewindow(w::AbstractWindow) 
    GLFW.SetWindowShouldClose(to_native(w.screen), true)
end
function window_isopen(w::AbstractWindow)
    return events(w.f).window_open
end
function cleanup(w::AbstractWindow)
    close(w.timer)
    # closewindow(w)
    # delete!(getml().windowlist, w)
    delete!(getwindows(), w)
end

Base.getindex(aw::AbstractWindow) = getindex(aw.other)
Base.setindex!(aw::AbstractWindow, v) = setindex!(aw.other, v)

include("AnalysisWindow.jl")
include("AvgWindow.jl")
include("LayerWindow.jl")



function LayerAxis(gridposition, l; colormap = :thermal, kwargs...)
    ax = Axis(gridposition, aspect = 1; kwargs...)
    ax.yreversed = @load_preference("makie_y_flip", default = false)
    img_ob = Observable(state(l))
    im = image!(ax, img_ob, fxaa = false, interpolate = false; colormap)
    im.colorrange[] = stateset(l)
    return ax, im, img_ob
end

function empty_window()
    f = Figure();
    newscreen = GLMakie.Screen()
    display(newscreen, f)
    return f, newscreen, window_open(f)
end
export empty_window

function create_window_timer(update, isopen; interval = 1/60)
    timer = PTimer((timer) -> update(), 0., interval = interval)
    on(isopen) do x
        if !x
            close(timer)
        end
    end
    return timer
end

function window_open(f::Figure)
    return events(f).window_open
end
export window_open