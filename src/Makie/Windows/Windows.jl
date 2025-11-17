abstract type AbstractWindow end
const windows = Dict{UUID,AbstractWindow}()
mutable struct MakieWindow{type, O} <: AbstractWindow
    uuid::UUID
    f::Figure
    screen::GLMakie.Screen

    maintimer::PTimer
    mainupdate::Tuple{Vararg{Function}}

    timers::Vector{PTimer} # Includes the polling timer
    polled_observables::Vector{PolledObservable} # Observables that are polled,
                                                    # all at some interval set at the beginning

    funcs::NamedTuple # TODO: What's this for?
    other::Dict{Symbol, Any}
    obj_ptr::O
end

MakieWindow(u,f,mt,mu,s,t,po,fs,o,op = nothing) = MakieWindow{:Any, typeof(op)}(u,f,mt,mu,s,t,po,fs,o,op)


function object(mw::MakieWindow)
    mw.obj_ptr
end

getinterval(mw::MakieWindow) = getinterval(mw.maintimer) # Main update interval
getdelay(mw::MakieWindow) = getdelay(mw.maintimer)

function newmaintimer!(w::MakieWindow)
    close(w.maintimer)
    _newmaintimer!(w, w.mainupdate)
end

function _newmaintimer!(w::MakieWindow, mainupdate::Tuple{Vararg{Function}})
    w.maintimer = PTimer((timer) -> begin
        func_tuple_unroll(mainupdate, tuple(w))
    end, getdelay(w), interval = getinterval(w))
end

function pushmainfunc!(w::MakieWindow, func::Function)
    w.mainupdate = (w.mainupdate..., func)
    newmaintimer!(w)  # Rebuild the maintimer with the new function
end


function pushpolled!(w::MakieWindow, po::PolledObservable)
    if isempty(w.polled_observables) #  If this is the first polled observable, create a timer to poll
        push!(w.timers, PTimer((timer) -> poll!.(w.polled_observables), 0., interval = 1/w[:polling_rate]))
    end
    push!(w.polled_observables, po)
end

function Base.close(window::MakieWindow)
    # println("Closing window ", window.uuid)
    close(window.maintimer)
    # println("Closed main timer")
    close.(window.timers)
    # println("Closed other timers")
    close.(window.polled_observables)
    # println("Closed polled observables")
    close_window(window)
    # println("Called custom close_window")
    # Only close GLFW if window is still open (avoid circular callback)
    if window[:window_open][]
        close_glfw(window)
    end
    # println("Closed GLFW window")
end

function close_window(window::MakieWindow) #Overloadable for custom closing
    return
end


"""
Creates a new Makie window with a figure and a screen
It also registers the window in the windows dictionary
"""
function new_window(;window_type = :Any, objectptr = nothing, refresh_rate = 30, polling_rate = 10, kwargs...)
    f, screen, window_open = empty_window(;kwargs...)
    u1 = uuid1()
    mainfuncs = tuple()
    maintimer = PTimer((timer) -> begin
        for f in mainfuncs
            f()
        end
    end, 0., interval = 1/refresh_rate)
    d = Dict{Symbol, Any}(:window_open => window_open)
    w = MakieWindow{window_type, typeof(objectptr)}(u1, f, screen, maintimer, mainfuncs, PTimer[], PolledObservable[], (;), d, objectptr)
    w[:polling_rate] = polling_rate

    w[:paused] = Observable(false)
    on(w[:paused]) do x
        if x
            close.(w.timers)
        else
            start.(w.timers)
        end
    end

    on(window_open) do x
        if !x
            # println("Window open set to false, closing window.")
            close(w)
            # println("Deleting window from registry.")
            delete!(windows, w)
            # println("Window deleted.")
        end
    end

    # CMD + W to close
    on(events(f.scene).keyboardbutton) do events
        hotkey = (Keyboard.left_super, Keyboard.w)
        if ispressed(f, hotkey)
            close(w)
        end
    end

    register_window(w, u1)
    return w
end


# Dict accessing
Base.getindex(w::MakieWindow, key) = w.other[key]
Base.setindex!(w::MakieWindow, val, key) = setindex!(w.other, val, key)
Processes.ispaused(w::MakieWindow) = w[:paused]::Observable{Bool}
togglepause(w::MakieWindow) = w[:paused][] = !w[:paused][]


function pushtimer!(w::MakieWindow, t::PTimer)
    if w[:paused][]
        pause(t)
    end
    push!(w.timers, t)
end

window_open(w::MakieWindow) = w.other[:window_open]::Observable{Bool}

# Is this type of window unique?
pushtype(::W) where W<:AbstractWindow = pushtype(W) 
function close_glfw(w::AbstractWindow) 
    GLFW.SetWindowShouldClose(to_native(w.screen), true)
end
function window_isopen(w::AbstractWindow)
    return events(w.f).window_open
end
function cleanup(w::AbstractWindow)
    close(w)
    # close_glfw(w)
    # delete!(getml().windowlist, w)
    delete!(getwindows(), w)
end

Base.getindex(aw::AbstractWindow) = getindex(aw.other)
Base.setindex!(aw::AbstractWindow, v) = setindex!(aw.other, v)

include("AnalysisWindow.jl")
include("AvgWindow.jl")
include("LayerWindow.jl")
include("Connections.jl")
include("LinesWindow.jl")


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



