using QML
using Observables
using CxxWrap
using ColorSchemes
using Images

# ENV["QSG_RENDER_LOOP"] = "basic"

# function imagesc(data::AbstractMatrix{<:Real};
#     colorscheme::ColorScheme=ColorSchemes.viridis,
#     maxsize::Integer=512, rangescale=:extrema)
# s = maximum(size(data))
# if s > maxsize
# return imagesc(imresize(data, ratio=maxsize/s);   # imresize from Images.jl
#         colorscheme, maxsize, rangescale)
# end
# return get(colorscheme, data, rangescale) # get(...) from ColorSchemes.jl
# end
# const img = Ref(imagesc(zeros(10,10)))

# qmlfile = joinpath(dirname(Base.source_path()), "test.qml")

# # loadqml( qmlfile, obs = sim.pmap)
# # loadqml(qmlfile)
# # exec()

# txt = Observable("txt")
# txt2 = Observable(2)

# const pap = JuliaPropertyMap(
#     "txt" => txt
# )

# function showlatest(buffer::Array{UInt32, 1}, width32::Int32, height32::Int32)
#     buffer = reshape(buffer, size(img[]))
#     buffer = reinterpret(ARGB32, buffer)
#     buffer .= transpose(img[])
#     return
# end

# export showlatest_cfunction
# showlatest_cfunction = CxxWrap.@safe_cfunction(showlatest, Cvoid, 
#                                                (Array{UInt32,1}, Int32, Int32))

# loadqml( qmlfile, showlatest = showlatest_cfunction)
# exec()

function showlatestg(buffer::Array{UInt32, 1}, width32::Int32, height32::Int32)
    buffer = reshape(buffer, size(img))
    buffer = reinterpret(ARGB32, buffer)
    buffer .= img
    return
end

showlatestexp = :(function showlatestl(buffer::Array{UInt32, 1}, width32::Int32, height32::Int32)
    buffer = reshape(buffer, size(img))
    buffer = reinterpret(ARGB32, buffer)
    buffer .= img
    return
end)

cexp = :((CxxWrap.CxxWrapCore).SafeCFunction($(Expr(:cfunction, Ptr{Nothing}, :(:showlatestexp), :Cvoid, :(Core.svec(Array{UInt32, 1}, Int32, Int32)), :(:ccall))), Cvoid, [Array{UInt32, 1}, Int32, Int32]))



function test()
    img = zeros(ARGB32,500,500) 
    
    
    # @cfunction(showlatestl, Cvoid,  (Array{UInt32,1}, Int32, Int32))
    # eval(:($(Expr(:cfunction, Ptr{Nothing}, :(:($showlatestl)), :Cvoid, :(Core.svec(Ref{Cuint}, Cint, Cint)), :(:ccall)))))

    :((CxxWrap.CxxWrapCore).SafeCFunction($(Expr(:cfunction, Ptr{Nothing}, :(:showlatestl), :Cvoid, :(Core.svec(Array{UInt32, 1}, Int32, Int32)), :(:ccall))), Cvoid, [Array{UInt32, 1}, Int32, Int32]))
    
end



function test2()
    eval($test())
end


function showlatesteval()
    function showlatest(buffer::Array{UInt32, 1}, width32::Int32, height32::Int32)
        buffer = reshape(buffer, size(img))
        buffer = reinterpret(ARGB32, buffer)
        buffer .= img
        return
    end

    @eval $:(CxxWrap.@safe_cfunction($showlatest, Cvoid, (Array{UInt32,1}, Int32, Int32)))
end

@cfunction($showlatestg, Cvoid,  (Array{UInt32,1}, Int32, Int32))
@cfunction(showlatestg, Cvoid, (Ref{Cuint},Cint,Cint) )

@macroexpand CxxWrap.@safe_cfunction(showlatest, Cvoid, 
                                                (Array{UInt32,1}, Int32, Int32))