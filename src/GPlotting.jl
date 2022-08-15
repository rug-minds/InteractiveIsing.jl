export gToImg, resizeGImg, saveGImg, imagesc, plotAdj

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

function resizeGImg(img,N,minsize)
    factor = Int32(floor(minsize/N))

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

function gToImg(g::AbstractIsingGraph, minsize = 500)
    tempimg = transpose(imagesc(reshape(g.state, (g.N,g.N) )  ))
    if g.N < minsize
        tempimg = resizeGImg(tempimg,g.N,minsize)
    end
    return tempimg
end

function saveGImg(g)
    foldername = imgFolder()
    println("Image saved to $(foldername)")
    save(File{format"PNG"}("$(foldername)/Ising Img $(nowStr()).PNG"), gToImg(g,500))
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
