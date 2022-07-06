# __precompile__()

# module GPlotting

# include("ising_graph.jl")

# using ColorSchemes

# export gToImg


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

function gToImg(g::AbstractIsingGraph, maxsize = 512)
    return imagesc(reshape(g.state, (g.N,g.N) )  , maxsize=maxsize  )
end

# end
