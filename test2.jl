using ColorSchemes
function newCirc(r::Integer)
    circpoints::Vector{Tuple} = []
    r2 = r^2
    for x in (-r):r
        for y in (-r):r
            if x^2+y^2 < r2
                push!(circpoints,tuple(y,x))
            end
        end
    end
    return circpoints
end             
# Make image from circle
function ordCircleToImg(r)
    mat = zeros(2*r-1,2*r-1)
    circ = newCirc(r)
    for point in circ
        mat[point[1] + r ,point[2] + r ] = 1
    end
    imagesc(mat)
end

function circleToImg(circ)
    head(tuple) = tuple[1]
    tail(tuple) = tuple[2]
    maxi = head(findmax(head.(circ)))
    maxj = head(findmax(tail.(circ)))

    mat = zeros(maxi,maxj)
    for point in circ
        mat[point[1] ,point[2]] = 1
    end
    imagesc(mat)
end

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