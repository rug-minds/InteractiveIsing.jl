#Converts matrix of reals into matrix of rgb data based on color scheme
function imagesc(data::AbstractMatrix{<:Real};
                 colorscheme::ColorScheme=ColorSchemes.viridis,
                 maxsize::Integer=512, rangescale=:extrema)
    s = maximum(size(data))
    if s > maxsize
        return imagesc(imresize(data, ratio=maxsize/s);   # imresize from Images.jl
                       colorscheme, maxsize, rangescale)
    end
    return get(colorscheme, data, rangescale) # get(...) from ColorSchemes.jl
end

#Display Ising IsingGraph
dispIsing(julia_display::JuliaDisplay, g) = let size = g.size
    if size <= 512
        display(julia_display, imagesc(permutedims( reshape(g.state, (g.N,g.N) ) ) , maxsize=g.N  ))
    else
        display(julia_display, imagesc(permutedims( reshape(g.state, (g.N,g.N) ) ) ) )
    end
end

#Display Ising IsingGraph
dispIsingREPL(g) = let size = g.size
    if size <= 512
        display(imagesc(permutedims( reshape(g.state, (g.N,g.N) ) ) , maxsize=g.N  ))
    else
        display(imagesc(permutedims( reshape(g.state, (g.N,g.N) ) ) ) )
    end
end



w"""
OLD PLOTTING FUNCTIONS
"""
function plotHeat(g, fig, plot_obj)
    plot_obj[1] = reshape(g.state, (g.N,g.N))
end

function loopPlotOld(g, loops, updatefunc , T, J)
    fig, ax , plot_obj = heatmap(reshape(g.state, (g.N,g.N)), colorrange=(-1, 1))
    display(fig)
    plotHeat(g,fig,plot_obj)
    @time for loop in 1:loops
        updatefunc(g.state,g.adj,T, J)
        plotHeat(g,fig,plot_obj)
    end
end

function loopPlot(julia_display::JuliaDisplay, g, loops, updatefunc , T, J)
    dispIsing(julia_display,g)
    for loop in 1:loops
        updatefunc(g.state,g.adj,T, J)
        dispIsing(julia_display,g)
    end
end


