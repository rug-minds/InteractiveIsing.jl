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
    w = axis_window(window_type = :Lines;kwargs...)
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

function get_data(w::MakieWindow{:Lines})
    return (x = w[:xs], y = w[:ys])
end
export get_data

function get_plot(w::MakieWindow{:Lines})
    display(w.f)
end
export get_plot


