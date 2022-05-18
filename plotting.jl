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

function gToImg(g::IsingGraph)
    return imagesc(reshape(g.state, (g.N,g.N) )  , maxsize=512  )
end

#Display Ising IsingGraph
dispIsing(julia_display::JuliaDisplay, g) = let size = g.size
    if size <= 512
        display(julia_display, imagesc(permutedims( reshape(g.state, (g.N,g.N) ) ) , maxsize=g.N  ))
    else
        display(julia_display, imagesc(permutedims( reshape(g.state, (g.N,g.N) ) ) ) )
    end
end