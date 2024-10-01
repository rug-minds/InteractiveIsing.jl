abstract type AbstractWindow end
const windows = Dict{UUID,AbstractWindow}()
mutable struct MakieWindow <: AbstractWindow
    uuid::UUID
    f::Figure
    screen::GLMakie.Screen
    timers::Vector{PTimer}
    other::Dict{Symbol, Any}
end

# Dict accessing
Base.getindex(w::MakieWindow, key) = w.other[key]
Base.setindex!(w::MakieWindow, val, key) = setindex!(w.other, val, key)
function pushtimer!(w::MakieWindow, t::PTimer)
    if w[:paused][]
        pause(t)
    end
    push!(w.timers, t)
end
window_open(w::MakieWindow) = w.other[:window_open]

# Is this type of window unique?
pushtype(::W) where W<:AbstractWindow = pushtype(W) 
function closewindow(w::AbstractWindow) 
    GLFW.SetWindowShouldClose(to_native(w.screen), true)
end
function window_isopen(w::AbstractWindow)
    return events(w.f).window_open
end

closetimers(w::MakieWindow) = close.(w.timers)

function cleanup(w::AbstractWindow)
    closetimers(w)
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

"""
Creates a new Makie window with a figure and a screen
It also registers the window in the windows dictionary
"""
function new_window(;kwargs...)
    f, screen, window_open = empty_window(;kwargs...)
    u1 = uuid1()
    w = MakieWindow(u1, f, screen, [], Dict(:window_open => window_open))
    w[:paused] = Observable(false)
    on(window_open) do x
        if !x
            closetimers(w)
            delete!(windows, w)
        end
    end

    # CMD + W to close
    on(events(f.scene).keyboardbutton) do events
        hotkey = (Keyboard.left_super, Keyboard.w)
        if ispressed(f, hotkey)
            closewindow(w)
        end
    end

    register_window(w, u1)
    return w
end

function axis_window(;pausebutton = true, kwargs...)
    w = new_window(;kwargs...)
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

function linesprocess(func, nrepeats, xtype = Float64, ytype = Float64)
    prepare = (proc, oldargs, newargs) -> (x = xtype[NaN] , y = ytype[NaN])
    return  makeprocess(func, nrepeats; prepare)
end
export linesprocess

"""
Create a new Makie window with a lines plot for the observables x and y
    Kwarg:
        fps: frames per second
"""
function lines_window(linesp; fps = 30, kwargs...)
    (;proc, x, y) = getargs(linesp)
    xref = Ref(x)
    yref = Ref(y)
    w = axis_window(;kwargs...)
    x = xref[]
    y = yref[]
    
    first = true
    w[:xs] = [x]
    w[:ys] = [y]

    xob = Observable(@view x[1:end])
    yob = Observable(@view y[1:end])

    lines = lines!(w[:ax], xob, yob)
    w[:lines] = [lines]

    function timerfunc(timer)
        minlength = min(length(xref[]),length(yref[]))
        xob.val = @view xref[][1:minlength]
        yob.val = @view yref[][1:minlength]
        notify(xob)
        autolimits!(w[:ax])
    end
    
    # pushtimer!(w,PTimer((timer) -> begin notify(x); autolimits!(w[:ax]); end, 0., interval = 1/fps))
    pushtimer!(w, PTimer(timerfunc, 0., interval = 1/fps))
    
    reset() = begin
        syncclose(proc)
        for line in w[:lines]
            delete!(w[:ax], line)
        end
        deleteat!(w[:lines], 1:length(w[:lines]))
        deleteat!(w[:xs], 1:length(w[:xs]))
        deleteat!(w[:ys], 1:length(w[:ys]))
        first = true
    end
    # Reset Button
    resetbutton = Button(w.f[0,1][1,2], label = "Reset", tellwidth = false, height = 28)
    on(resetbutton.clicks) do _
        reset()
    end

    function newlines!()
        syncclose(proc)
        xob = Observable(@view xref[][1:end-1])
        yob = Observable(@view yref[][1:end-1])
        createtask!(proc)
        (;x, y) = getargs(linesp)
        xref[] = x
        yref[] = y
        xob = Observable(@view xref[][1:end])
        yob = Observable(@view yref[][1:end])
        
        
        runtask!(proc)
        push!(w[:lines], lines!(w[:ax], xob, yob))
    end
    #Rerun Button
    rerunbutton = Button(w.f[0,1][1,3], label = "Rerun", tellwidth = false, height = 28)
    on(rerunbutton.clicks) do _
            if !first
                push!(w[:xs], xref[])
                push!(w[:ys], yref[])
            else
                first = false
            end

            start.(w.timers)
            w[:paused][] = false
            newlines!()
            unpause(proc)

        
    end

    # Space to reset
    on(events(w.f.scene).keyboardbutton) do event
        if event.action == Keyboard.press
            if event.key == Keyboard.space
                reset()
            end
        end
    end

    # If a proc is given, add the proc controls
    if !isnothing(proc)
        w[:proc] = proc
        on(window_open(w)) do x
            if !x
                quit(proc)
            end
        end

        on(w[:pausebutton].clicks) do _
            if w[:paused][]
                pause(proc)
            else
                unpause(proc)
            end
        end
    end

    runtask!(proc) 
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