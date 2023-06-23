function getword(stream::IOBuffer; ignore_comment = false)::String
    string = Char[]
    char = ' '
    while !eof(stream) && isspace(char)
        char = read(stream, Char)
        if ignore_comment && char == '#'
            while !eof(stream) && char != '\n'
                char = read(stream, Char)
            end
        end
    end
    
    while !eof(stream) && !isspace(char)
        push!(string, char)
        char = read(stream, Char)
    end

    return join(string)
end

#forward declaration of struct that reads all files in the same folder
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

            if range[1] >= 9 && filestr[mutablerange] == "mutable"
                range = mutablerange[1]:range[end]
            end

            fileidx = idx
            break
        end
    end

    file = read(files[fileidx], String)
    # add string until "end" is found
    # find next occurrence of "end" after range
    endrange = range
    stream = IOBuffer(file[1:end])
    stream.ptr = endrange[end]
    open_expressions = 1
    num_ends = 0
    open_expression_terms = ["function", "while", "for", "if", "begin", "let", "quote"]
    while open_expressions > num_ends || eof(stream)
        # Fix so that it also works         
        # word = readuntil(stream, ' ', keep = false)
        word = getword(stream, ignore_comment = true)
        if '#' ∈ word
            readline(stream)
        end 
        # word = strip(word)
        if word ∈ open_expression_terms
            open_expressions += 1
        elseif word == "end"
            num_ends += 1
        end
    end

    endrange = stream.ptr - 1

    str = file[range[1]:endrange[end]]
    return str

end 

#macro defines a struct with the given name in a try catch block
# with a field that has a type that's gona make it fail
macro ForwardDeclare(structname, subfolder)
    #get files from current dir
    files = readdir(joinpath(@__DIR__, subfolder))

    # get all files that end in .jl
    files = filter(x -> endswith(x, ".jl"), files)

    # add the path to the files
    files = map(x -> joinpath(@__DIR__, subfolder, x), files)
    structstring = getstruct(string(structname), files)
    # println(structstring)
    # println("Searching Dir $(@__DIR__)/$subfolder for struct $structname")
    expr = Meta.parse(structstring)
    return esc(quote
                try
                    $expr
                catch
                end
            end)
end

abstract type AbstractIsingGraph end

@ForwardDeclare IsingLayer ""

function testfunc(il::IsingLayer)
    println("testfunc")
end
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

