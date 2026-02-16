abstract type AbstractWindow end

# """
# Get data in the window
# """
# Base.getindex(aw::AbstractWindow) = getindex(aw.other)
# """
# Set data in the window
# """
# Base.setindex!(aw::AbstractWindow, v) = setindex!(aw.other, v)


const windows = Dict{UUID,AbstractWindow}()


"""
Window struct for the Makie interface
    Contains the figure, screen, timers, and other data related to the window

    Maintimer is a pausable timer that runs iterates over mainupdate functions
        mainupdate is a tuple of functions that are called every time the maintimer ticks
        like a notify function for the observables for the interface

    Timers is a vector of other timers that can be paused and closed when the window is closed

    If a function is pushed to the mainupdate tuple, the maintimer is closed, 
        and then rebuilt with the new functions for type stablility

"""
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

# Dict accessing
Base.getindex(w::MakieWindow, key) = w.other[key]
Base.setindex!(w::MakieWindow, val, key) = setindex!(w.other, val, key)
Processes.ispaused(w::MakieWindow) = w[:paused]::Observable{Bool}
togglepause(w::MakieWindow) = w[:paused][] = !w[:paused][]

"""
Get updating interval of the main timer
"""
getinterval(mw::MakieWindow) = getinterval(mw.maintimer) # Main update interval
"""
Get delay of the main timer
"""
getdelay(mw::MakieWindow) = getdelay(mw.maintimer)

"""
Set new maintimer
"""
function newmaintimer!(w::MakieWindow)
    close(w.maintimer)
    _newmaintimer!(w, w.mainupdate)
end

"""
Rebuild the maintimer with a new set of functions
"""
function _newmaintimer!(w::MakieWindow, mainupdate::Tuple{Vararg{Function}})
    w.maintimer = PTimer((timer) -> begin
        func_tuple_unroll(mainupdate, tuple(w))
    end, getdelay(w), interval = getinterval(w))
end

"""
Push a function to the mainupdate tuple and rebuild the maintimer
"""
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

"""
Push a timer to the window's timers vector and pause it if the window is currently paused
"""
function pushtimer!(w::MakieWindow, t::PTimer)
    if w[:paused][]
        pause(t)
    end
    push!(w.timers, t)
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
    close_glfw(window)
    # println("Closed GLFW window")
end

function close_window(window::MakieWindow) #Overloadable for custom closing
    return
end


"""
Creates a new Makie window with a figure and a screen
It also registers the window in the windows dictionary
"""
function new_window(;window_type = :Any, objectptr = nothing, reprepare_rate = 30, polling_rate = 10, kwargs...)
    f, screen, window_open = empty_window(;kwargs...)
    u1 = uuid1()
    mainfuncs = tuple()
    maintimer = PTimer((timer) -> begin
        for f in mainfuncs
            f()
        end
    end, 0., interval = 1/reprepare_rate)
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



window_open(w::MakieWindow) = w.other[:window_open]::Observable{Bool}

# Is this type of window unique?
pushtype(::W) where W<:AbstractWindow = pushtype(W) 

"""
Close the window by setting the GLFW should close flag to true
"""
function close_glfw(w::AbstractWindow) 
    GLFW.SetWindowShouldClose(to_native(w.screen), true)
end
"""
Window open observable
"""
function window_isopen(w::AbstractWindow)
    return events(w.f).window_open
end

"""
Cleanup function to close the window and delete it from the registry
"""
function cleanup(w::AbstractWindow)
    close(w)

    delete!(getwindows(), w)
end