struct WindowList
    d::Dict{DataType, Vector{AbstractWindow}}
    pushtype::Dict{DataType, Symbol}
end

WindowList() = WindowList(Dict{DataType, Vector{AbstractWindow}}(), Dict{DataType, Symbol}())

function Base.push!(wl::WindowList, item)
    _pushtype = pushtype(item)
    if _pushtype == :unique
        ws = get!(wl.d, typeof(item), AbstractWindow[])
        if isempty(ws)
            push!(ws, item)
        else
            if !(ws[1].screen === item.screen)
                closewindow(ws[1])
            end
            ws[1] = item
        end
    else # _pushtype == :multiple
        push!(get!(wl.d, typeof(item), AbstractWindow[]), item)
    end
end

function Base.getindex(wl::WindowList, t::DataType)
    if pushtype(t) == :unique
        return get(wl.d, t, AbstractWindow[])[1]
    else
        return get(wl.d, t, AbstractWindow[])
    end
end

function findidx(wl::WindowList, t::AbstractWindow)
    typof(t) => findfirst(get(wl.d,typeof(t),AbstractWindow[]), t)
end

function Base.getindex(wl::WindowList, p::Pair)
    if isnothing(p.second)
        return AbstractWindow[]
    end
    wlist = get!(wl.d, p.first, AbstractWindow[])
    if !isempty(wlist)
        return wlist[p.second]
    else
        return AbstractWindow[]
    end
end

function Base.delete!(wl::WindowList, w::AbstractWindow)
    windowidx = findidx(wl, w)
    delete!(wl, windowidx)
    return wl
end

function Base.delete!(wl::WindowList, p::Pair)
    if isnothing(p.second) # if not found do nothing
        return wl
    end
    wlist = get!(wl.d, p.first, AbstractWindow[])
    w = wlist[p.second]
    cleanup(w)
    deleteat!(wlist, p.second)
end