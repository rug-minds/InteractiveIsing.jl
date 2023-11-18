layer(l::IsingLayer) = l.l
layer(g::IsingGraph) = convert(IsingLayer, g)
function createAnalysisWindow(l, panels...;kwargs...)
    AnalysisWindow(layer(l), panels...;kwargs...)
end

    

export createAnalysisWindow

mutable struct AnalysisWindow <: AbstractWindow
    l::IsingLayer    
    f::Figure
    screen::GLMakie.Screen
    other::Dict{String, Any}
    shared_funcs::Dict{String,Tuple{Function,Function,Function, Any}} #update, reset, pause, data
    obsfuncs::Vector{Any}
    running::Observable{Bool}
    timer::PTimer
    AnalysisWindow(a,b,c,d, t = EmptyPTimer()) = new(a,b,c,d,Dict(),[], Observable(true), t)
end
Base.getindex(aw::AnalysisWindow, i) = aw.other[i]
Base.setindex!(aw::AnalysisWindow, v, i) = aw.other[i] = v
pushtype(::Type{AnalysisWindow}) = :unique
Base.get(aw::AnalysisWindow, i, default) = get(aw.other, i, default)
Base.get!(aw::AnalysisWindow, i, default) = get!(aw.other, i, default)
Base.delete!(aw::AnalysisWindow, i) = delete!(aw.other, i)

function createfuncs(shared_funcs)
    if isempty(shared_funcs)
        return (), (), ()
    end
    updates = tuple([x[1] for x in values(shared_funcs)]...)
    resets = tuple([x[2] for x in values(shared_funcs)]...)
    pauses = tuple([x[3] for x in values(shared_funcs)]...)
    return updates, resets, pauses
end

function broadcast_args(fs, args...)
    # println("Broadcasting args")
    # println("Resettting fs: $fs")
    # println("With args: $args")
    for f in fs
        f(args...)
    end
end

export AnalysisWindow
# You can delete a plot object directly via delete!( ax, plotobj) . You can also remove all plots with empty!( ax) .
function AnalysisWindow(l::IsingLayer, panel1, panel2 = correlation_panel; shared_interval = 1/30)
    f, screen, isopen = empty_window()

    analysiswindow = AnalysisWindow(l, f, screen, Dict{String, Any}())
    axesgrid = GridLayout(f[1:3,1])

    # Create panel 1
    isactive1, reset1, pause1, axis1 = panel1(analysiswindow, axesgrid, (2,1), l)
    # Create panel 2
    isactive2, reset2, pause2, axis2 = panel2(analysiswindow, axesgrid, (3,1), l)
    

    resetfuncs = Function[]
    ####
    # AFTER PANELS ARE SET, RUN THE SHARED TimedFunctions
    updates, resets, pauses = createfuncs(analysiswindow.shared_funcs)


    if !isnothing(updates)
        analysiswindow.timer = PTimer((timer) -> begin
            broadcast_args(updates)
        end, 0., interval = shared_interval)
    end
    


    ## OTHER ELEMENTS
    buttongrid = GridLayout(axesgrid[1,1])
        println("Resets: $resets")
        ### RESET BUTTON
        # Button with text "Reset"
            b1 = Button(buttongrid[1,1], label = "Reset graph", tellwidth = false, tellheight = false)
            on(b1.clicks) do x
                reset1()
                reset2()
                broadcast_args(resets)
            end

        ### PAUSEBUTTON
            running = analysiswindow.running
            pausebuttonlabel = lift((x) -> x ? "Pause" : "Resume", running)
            b2 = Button(buttongrid[1,2], label = pausebuttonlabel, tellwidth = false, tellheight = false)
            on(b2.clicks) do x
                running[] = !running[]
            end

            on(running) do x
                pause1(x)
                pause2(x)
                broadcast_args(pauses, x)
                if !x
                    close(analysiswindow.timer)
                else 
                    start(analysiswindow.timer)
                end
            end
            rowsize!(axesgrid, 1, 40)

        # RESET STATE BUTTON
        # Button to set state to 1
            b2 = Button(buttongrid[1,3], label = "Set state", tellwidth = false, tellheight = false)
            on(b2.clicks) do x
                state(l) .= 1
            end

        #### TEMPERATURE SLIDER
        # Tempslider Permanent
            templabel = lift((x) -> "T $x", temp(sim(graph(l))))
            push!(analysiswindow.obsfuncs, templabel.inputs[1])
            slidergrid = GridLayout(axesgrid[2,2])
            rowsize!(slidergrid, 1, 20)
            sliderlabel = Label(slidergrid[1,1], templabel, tellwidth = false, tellheight = false)
            tempslider = Slider(slidergrid[2,1], horizontal = false, range = 0:0.1:5)
            tempslider.value[] = temp(graph(l))
            
            on(tempslider.value) do x
                settemp(graph(l), x)
            end
    
    # When window is closed
    on(window_open(f)) do _
        isactive1[] = false
        isactive2[] = false
        cleanup(analysiswindow)
    end

    # ml["analysisWindow"] = analysiswindow
    push!(windowlist, analysiswindow)

    return analysiswindow

end


function cleanup(awindow::AnalysisWindow)
    close(awindow.timer)
    if !isempty(awindow.obsfuncs)
        off.(awindow.obsfuncs)
    end
    return nothing
end



####------------####
######-PANELS-######
####------------####

# Panel for the temperature phase transition
export MT_panel
function MT_panel(window, axesgrid, pos, layer)
    etype = eltype(layer)    

    t_buffer = Observable(CircularBuffer{etype}(600))
    m_buffer = Observable(CircularBuffer{etype}(600))
    m_avg = AverageCircular(etype,10)
        
    window["t_buffer"] = t_buffer
    window["m_buffer"] = m_buffer

   
    axis = Axis(getindex(axesgrid,pos...), tellwidth = false, tellheight = false)
    lns = lines!(axis, t_buffer, m_buffer, color = :blue)
    xlims!(axis, -0.1, 5)
    ylims!(axis, -0.1, 1.1)

    function reset()
        empty!(t_buffer[])
        empty!(m_buffer[])
    end

    function update()
        push!(m_avg, sum(state(layer))/nstates(layer))
        push!(t_buffer[], temp(graph(layer)))
        push!(m_buffer[], abs(avg(m_avg)))

        notify(t_buffer)
        notify(m_buffer)
        reset_limits!(axis)
    end
    timer = PTimer((timer) -> update(), 0., interval = 1/30)

    isactive = Observable(true)
    on(isactive) do x
        if !x
            close(timer)
        end
    end

    function pause(running)
        if !running
            close(timer)
        else
            start(timer)
        end
    end
    return isactive, reset, pause, lns
end


# Panel for the magnetic field phase transition
export MB_panel
function MB_panel(window, axesgrid, pos, layer)
    etype = eltype(layer)    
    mbgrid = GridLayout(getindex(axesgrid,pos...))
    
    b_buffer = Observable(CircularBuffer{etype}(2400))
    m_buffer = Observable(CircularBuffer{etype}(2400))
    m_avg = AverageCircular(etype,10)

    x_left = -4
    x_right = 4

    slider = Slider(mbgrid[1,2], range = x_left:0.01:x_right, horizontal = false, tellwidth = false, tellheight = false)
    on(slider.value) do x
        globalB!(layer, x)
    end

    axis = Axis(mbgrid[1,1], tellwidth = false, tellheight = false)
    xlims!(axis, x_left, x_right)
    ylims!(axis, -1.1, 1.1)
    lines = lines!(axis, b_buffer, m_buffer, color = :blue)
    colsize!(mbgrid, 2, 40)
    
    function reset()
        empty!(b_buffer[])
        empty!(m_buffer[])
    end

    function update(timer)
        push!(m_avg, sum(state(layer))/nstates(layer))
        push!(b_buffer[], slider.value[])
        push!(m_buffer[], (avg(m_avg)))

        notify(b_buffer)
        notify(m_buffer)
    end

    timer = PTimer(update, 0., interval = 1/30)
    isactive = Observable(true)
    on(isactive) do x
        if !x
            close(timer)
        end
    end
    function pause(running)
        if !running
            close(timer)
        else
            start(timer)
        end
    end

    return isactive, reset, pause, axis
end

# Panel for the order parameter phase transition
function Tξ_panel()

end
using Statistics
## SHARED DATA
function shareddata_STDev(window)
    identifier = "shareddata_STDev"
    if haskey(window.shared_funcs, identifier)
        return window.shared_funcs[identifier][4] #Return the data
    end

    #Else create the data and update func
    layer = window.l

    data = stddata(Int[round(Int,abs(sum(state(layer))))])

    function update()
        push!(data, round(Int,abs(sum(state(layer)))) )
    end

    function reset()
        deleteat!(data, 2:length(data))
        data[1] = round(Int,abs(sum(state(layer))))
    end
    function pause(running)
        if running
            reset()
        end
    end

    obsfunc = on(temp(sim(graph(window.l)))) do _
        window.running[] = false
        # reset()
    end

    push!(window.obsfuncs, obsfunc)

    window.shared_funcs[identifier] = (update,reset,pause,data)

    return data
end

export Tχ_panel
function Tχ_panel(window, axesgrid, pos, layer)
    axis = Axis(getindex(axesgrid,pos...), tellwidth = false, tellheight = false)
    etype = eltype(layer)
    data = shareddata_STDev(window)
    window["data"] = data
    trange = 1:0.1:5
    # allts = [trange;]
    ts = Observable([temp(graph(layer))])


    χdata = Observable(zeros(etype, 1))

    scatter!(axis, ts, χdata, color = :blue, tellwidth = false)
    reset_limits!(axis)

    function reset()
        χdata.val = [temp(graph(layer))]
        ts.val = [temp(graph(layer))]
        notify(χdata)
        reset_limits!(axis)
    end

    function update(timer)
        t = temp(graph(layer))
        χdata[][end] = 1/t*std(data)
        notify(χdata)
        reset_limits!(axis)
    end

    timer = PTimer(update, 0., interval = 1/4)
    isactive = Observable(true)
    on(isactive) do x
        if !x
            close(timer)
        end
    end
    function pause(running)
        if !running
            close(timer)
        else
            push!(ts[], temp(graph(layer)))
            push!(χdata[],0)
            notify(χdata)
            start(timer)
        end
    end

    return isactive, reset, pause, axis
end


# Panel for the magnetic susceptibility phase transition
# Bar Plot
export χₘ_panel
function χₘ_panel(window, axesgrid, pos, layer)
    etype = eltype(layer)
    data = shareddata_STDev(window)
    push!(data, abs(sum(state(layer))))
    Mbars = Observable(data)
    
    # Label(box[2,1], st_dev; valign = :top, halign = :right, tellwidth = false, tellheight = false)
    axis = Axis(getindex(axesgrid,pos...), tellwidth = false, tellheight = false)
    histo = hist!(axis, Mbars,; bins = 100, color = :blue, tellwidth = false)


    timer = PTimer((timer) -> begin notify(Mbars); reset_limits!(axis) end, 0., interval = 1/10)
    isactive = Observable(true)
    
    window["Mbars"] = Mbars
    

    on(isactive) do x
        if !x
            close(timer)
            # delete!(histo)
        end
    end

    function reset()
        notify(Mbars)
    end

    function pause(running)
        if !running
            close(timer)
        else
            start(timer)
        end
    end

    return isactive, reset, pause, axis
end

function correlation_panel(window, axesgrid, pos, layer)
    etype = eltype(layer)
    corr_l, corr_val = correlationLength(layer)
    corr_r = Observable(corr_l)
    corr = Observable(corr_val)
    corr_avgs = [AverageCircular(etype, 10) for _ in 1:length(corr_val)]

    axis = Axis(getindex(axesgrid,pos...), tellwidth = false)
    correlation = lines!(axis, corr_r, corr, color = :blue, tellwidth = false)
    
    isactive = Observable(true)
    
    window["corr_r"] = corr_r
    window["corr"] = corr
    window["corr_avgs"] = corr_avgs

    function update(timer)
        # corr_l, corr_val = fetch(Threads.@spawn correlationLength(layer))
        corr_l, corr_val = correlationLength(layer)
    
        for (val_idx, val) in enumerate(corr_val)
            push!(corr_avgs[val_idx], val)
            corr[][val_idx] = avg(corr_avgs[val_idx])
        end
        notify(corr)
        reset_limits!(axis)
    end
    timer = PTimer(update, 0., interval = 1/10)

    on(isactive) do x
        if !x
            close(timer)
            delete!(window, "corr_r")
            delete!(window, "corr")
            delete!(window, "corr_avgs")
        end
    end

    function reset()
        corr[] = (correlationLength(layer))[2]
    end

    function pause(running)
        if !running
            close(timer)
        else
            start(timer)
        end
    end

    return isactive, reset, pause, axis
end

struct stddata{T} <: AbstractVector{T}
    d::Vector{T}
end

Base.length(d::stddata) = length(d.d)
Base.getindex(d::stddata, i) = d.d[i]
Base.setindex!(d::stddata, v, i) = d.d[i] = v
Base.push!(d::stddata, v) = push!(d.d, v)
Base.eltype(d::stddata) = eltype(d.d)
Base.iterate(d::stddata, state = 1) = iterate(d.d, state)
Base.isempty(d::stddata) = isempty(d.d)
Base.eachindex(d::stddata) = eachindex(d.d)
Base.deleteat!(d::stddata, i) = deleteat!(d.d, i)
Base.size(d::stddata) = size(d.d)

function Statistics.std(d::stddata)
    if isempty(d)
        return 0.
    end
    return std(d.d)
end
    
