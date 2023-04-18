# macro that takes a description of a struct and fowards declares it by putting a try catch block around it
# macro ForwardDeclare(strct)
#     return quote
#         try
#             $(esc(strct))
#         catch
#         end
#     end
# end
#read all files and find the struct that matches the name in strctname
function getstruct(strctname, files)
    # for every file look for the word struct
    range = nothing
    fileidx = 1
    for (idx, file) in enumerate(files)
        # read the file
        filestr = read(file, String)
        # find the index of the word struct
        range = findfirst("struct $strctname", filestr)

        

        # if it's found break the loop
        if range != nothing
            mutablerange = (range[1]- 8):(range[1]-2)

            if range[1] >= 9 && filestr[(range[1]- 8):(range[1]-1)] == "mutable"
                println("is mutable")
                range = (range[1]-8):range[end]
            end

            fileidx = idx
            break
        end
    end

    file = read(files[fileidx], String)
    # add string until "end" is found
    # find next occurrence of "end" after range
    endrange = findnext("end", file, range[end])

    str = file[range[1]:endrange[end]]

    return str

end

#macro defines a struct with the given name in a try catch block
# with a field that has a type that's gona make it fail
macro ForwardDeclare(structname)
    #get files from current dir
    files = readdir(@__DIR__)

    # get all files that end in .jl
    files = filter(x -> endswith(x, ".jl"), files)

    # add the path to the files
    files = map(x -> joinpath(@__DIR__, x), files)

    expr = Meta.parse(getstruct(string(structname), files))
   
    return esc(quote
                try
                    $expr
                catch
                end
            end)
end

abstract type AbstractIsingGraph end

@ForwardDeclare IsingLayer

struct IsingGraph <: AbstractIsingGraph
    layers::Vector{IsingLayer}
    d::Any
end

mutable struct IsingLayer <: AbstractIsingGraph
    g::IsingGraph
    d::Any

    IsingLayer() = (l = new())
end

struct SomeType
    layer::IsingLayer
    graph::IsingGraph
end

