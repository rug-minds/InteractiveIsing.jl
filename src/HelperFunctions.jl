#=Grid lattice functions=#

# Matrix Coordinates to vector Coordinates
@inline function coordToIdx(i, j, N)
    return (i - 1) * N + j
end

# Insert coordinates as tuple
coordToIdx((i, j), N) = coordToIdx(i, j, N)

# Go from idx to lattice coordinates
@inline function idxToCoord(idx::Integer, N::Integer)
    return ((idx - 1) ÷ N + 1, (idx - 1) % N + 1)
end

# Put a lattice index (i or j) back onto lattice by looping it around
function latmod(i, N)
    if i < 1 || i > N
        return i = mod((i - 1), N) + 1
    end
    return i
end

# Array functions

# Searches backwards from idx in list and removes item
# This is because spin idx can only be before it's own index in aliveList
function revRemoveSpin!(list,spin_idx)
    init = min(spin_idx, length(list)) #Initial search index
    for offset in 0:(init-1)
        @inbounds if list[init-offset] == spin_idx
            deleteat!(list,init-offset)
            return init-offset # Returns index where element was found
        end
    end
end

# Zip together two ordered lists into a new ordered list; fast  
function zipOrderedLists(vec1::Vector{T},vec2::Vector{T}) where T
    # result::Vector{T} = zeros(length(vec1)+length(vec2))
    result = Vector{T}(undef, length(vec1)+length(vec2))

    ofs1 = 1
    ofs2 = 1
    while ofs1 <= length(vec1) && ofs2 <= length(vec2)
        @inbounds el1 = vec1[ofs1]
        @inbounds el2 = vec2[ofs2]
        if el1 < el2
            @inbounds result[ofs1+ofs2-1] = el1
            ofs1 += 1
        else
            @inbounds result[ofs1+ofs2-1] = el2
            ofs2 += 1
        end
    end

    if ofs1 <= length(vec1)
        @inbounds result[ofs1+ofs2-1:end] = vec1[ofs1:end]
    else
        @inbounds result[ofs1+ofs2-1:end] = vec2[ofs2:end]
    end
    return result
end

# Deletes els from vec which should be ordered
# Assumes that els are in vec!
# Should I make ordered vec a type?
function remOrdEls(vec::Vector{T}, els::Vector{T}) where T
    # result::Vector{T} = zeros(length(vec)-length(els))
    result = Vector{T}(undef, length(vec)-length(els))
    it_idx = 1
    num_del = 0
    for el in els
            while el != vec[it_idx]
        
            result[it_idx - num_del] = vec[it_idx]
            it_idx +=1
        end
            num_del +=1
            it_idx += 1
    end
        result[(it_idx - num_del):end] = vec[it_idx:end]
    return result
end

# Remove first element equal to el and returns correpsonding index
function removeFirst!(list,el)
    for (idx,item) in enumerate(list)
        if item == el
            deleteat!(list,idx)
            return idx
        end
    end
end

#=
Threads
=#

# Spawn a new thread for a function, but only if no thread for that function was already created
# The function is "locked" using a reference to a Boolean value: spawned
function spawnOne(f::Function, spawned::Ref{Bool}, threadname = "", args... )
    # Run function, when thread is finished mark it as not running
    function runOne(func::Function, spawned::Ref{Bool}, args...)
        func(args...)
        spawned[] = false
        GC.safepoint()
    end

    # Mark as running, then spawn thread
    if !spawned[]
        spawned[] = true
        # Threads.@spawn runOne(f,spawned)
        runOne(f,spawned, args...)
    else
        println("Already spawned" * threadname)
    end
end

macro includefile(filepath)
    esc(Meta.parse(read(open(eval(filepath)), String)))
end

macro includefile(includepath, filename::String)
    esc(Meta.parse(read(open(eval(includepath) * "/" * filename), String)))
end

macro includetextfile(pathsymbs...)
    path = joinpath(@__DIR__,"textfiles", string.(pathsymbs)...)
    esc(Meta.parse(read(open(path), String)))
end

function setTuple(tuple, idx, val)
    if idx == 1
        return (val, tuple[2:end]...)
    elseif idx == length(tuple)
        return (tuple[1:(end-1)]..., val)
    else
        return (tuple[1:(idx-1)]...,val,tuple[(idx+1):end]...)
    end
end
export setTuple


"""
Registers the fieldnames of some struct B as methods for struct a
where 
    struct A
        b::B
        ...
    end

    struct B
        a
        b
        c
    end

    s.t.

    a = A(...)

    a(a) = a.b.a
    b(a) = a.b.b
    ...

Capitalization and naming has to be done like the above

    Also defines the same methods with an extra argument which are setters
"""
macro forward(Outer, Inner)
    funcs = quote end
    for name in fieldnames(eval(Inner))
        outervarname = lowercase(string(nameof(eval(Outer))))
        innervarname = lowercase(string(nameof(eval(Inner))))
        getstr = "@inline $name($outervarname) = $outervarname.$innervarname.$(string(name))"

        setstr = "@inline $name($outervarname, val) = $outervarname.$innervarname.$(string(name)) = val"
        push!(funcs.args, Meta.parse(getstr))
        push!(funcs.args, Meta.parse(setstr))
        push!(funcs.args, Meta.parse("export $name"))
        
    end
    return esc(funcs)
end


# Etc
# Not used?
function sortedPair(idx1::Integer,idx2::Integer):: Pair{Integer,Integer}
    if idx1 < idx2
        return idx1 => idx2
    else
        return idx2 => idx1
    end
end