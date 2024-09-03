using UUIDs
abstract type AbstractWindow end
const windows = Dict{UUID,AbstractWindow}()
struct MakieWindow <: AbstractWindow
    uuid::UUID
    f::Figure
    screen::GLMakie.Screen
    timers::Vector{PTimer}
    other::Dict{String, Any}
end

# Dict accessing
Base.getindex(w::MakieWindow, key) = w.other[key]
Base.setindex!(w::MakieWindow, val, key) = setindex!(w.other, val, key)

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
include("Connections.jl")



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

function new_window(;kwargs...)
    f, screen, window_open = empty_window(;kwargs...)
    u1 = uuid1()
    w = MakieWindow(u1, f, screen, [], Dict("window_open" => window_open))
    on(window_open) do x
        if !x
            close.(w.timers)
            delete!(windows, w)
        end
    end
    register_window(w, u1)
    return w
end
export new_window

function register_window(w::AbstractWindow, u::UUID)
    windows[u] = w
end

function create_window_timer(update, isopen; interval = 1/60)
    timer = PTimer((timer) -> update(), 0., interval = interval)
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