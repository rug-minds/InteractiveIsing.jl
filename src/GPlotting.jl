export gToImg, resizeGImg, saveGImg, imagesc, plotAdj

#Converts matrix of reals into matrix of rgb data based on color scheme
function imagesc(data::AbstractMatrix{<:Real};
                # colorscheme::ColorScheme=ColorSchemes.viridis,
                colorscheme::ColorScheme=ColorSchemes.viridis,
                maxsize::Integer=512, rangescale=:extrema)
    s = maximum(size(data))
    if s > maxsize
        return imagesc(imresize(data, ratio=maxsize/s);   # imresize from Images.jl
                    colorscheme, maxsize, rangescale)
    end
    return get(colorscheme, data, rangescale) # get(...) from ColorSchemes.jl
end

# Resizes image of graph by duplicating pixels by integer amount
# Determines integer factor based on specifying some max size
# N = image size (why not read this from img????)
# Maxsize specifies maxsize
function resizeGImg(img,N,maxsize)
    factor = Int32(floor(maxsize/N))

    newN = (N*factor)
    # new_size = trunc.(Int, (size,size) .* factor)
    # println(new_size)
    newimg = Matrix{RGB{Float64}}(undef,newN,newN)
    
    slices = 1:factor:newN

    step = factor-1
    for (vidx,vert_slice) in enumerate(slices[1:end])
        for (hidx,hori_slice) in enumerate(slices[1:end])
            newimg[vert_slice:(vert_slice+step), hori_slice:(hori_slice+step)] .= img[vidx,hidx]
        end
    end
    
    return newimg
end

# function gToImg(g::AbstractIsingGraph, maxsize = 600; colorscheme = ColorSchemes.viridis)
#     tempimg = imagesc(reshape(state(g), glength(g), gwidth(g)); colorscheme)
#     if gwidth(g) < maxsize
#         tempimg = resizeGImg(tempimg,gwidth(g),maxsize)
#     end
#     return tempimg
# end

function gToImg(g::AbstractIsingGraph; colorscheme = ColorSchemes.viridis )
    tempimg = imagesc(reshape(state(g), glength(g), gwidth(g)), maxsize = 2000; colorscheme)
    return tempimg
end
export gToImg

function checkImgSize(sim, layer, glength, gwidth, qmllen, qmlwid)
    if glength != qmllen || gwidth != qmlwid
        image(sim)[] = gToImg(layer, colorscheme = colorscheme(sim))
        qmllength(sim)[] = glength
        qmlwidth(sim)[] = gwidth
    end
end


function saveGImg(layers...)
    for layer in layers
        foldername = imgFolder()
        println("Image saved to $(foldername)")
        save(File{format"PNG"}("$(foldername)/Ising Img $(nowStr()).PNG"), gToImg(layer))
    end
end

function imgFolder()
    try; mkdir(joinpath(dirname(Base.source_path()),"Data")); catch end
    try; mkdir(joinpath(dirname(Base.source_path()), "Data", "Images")); catch end
    return joinpath(dirname(Base.source_path()), "Data", "Images")
end

function nowStr()
    nowtime = replace("$(now())"[3:(end)], "T" => " ")
    return nowtime
end

function plotAdj(g)
    imagesc(adjToMatrix(g.adj))
end
