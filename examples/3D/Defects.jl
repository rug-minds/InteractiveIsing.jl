using StaticArrays, SparseArrays, LoopVectorization, InteractiveIsing
import InteractiveIsing: AverageCircular
struct ConnectionAccessor
    spindx::Int
    outptr::UnitRange{Int32}
    outidx2ptr::SparseVector{Int32,Int32}
    inptr::Vector{Int32}
end

ConnectionAccessor(spindx, outptr) = ConnectionAccessor(spindx, outptr, Vector{Int32}(undef, length(outptr)), Int32[])

struct Accessors{eltype}
    data::Vector{ConnectionAccessor}
end

Base.@propagate_inbounds function Accessors(sp::SparseMatrixCSC)
    @assert size(sp, 1) == size(sp, 2) "Sparse matrix must be square"
    states = size(sp, 1)

    spvec_alloc = [([],[]) for _ in 1:states]
    inptr_alloc = [Int32[] for _ in 1:states]
    
    colval = 1
    for ptr in eachindex(sp.nzval)

        if ptr > sp.colptr[colval+1]-1
            colval += 1
        end

        out_spindx = sp.rowval[ptr]
        push!(inptr_alloc[out_spindx], ptr)

        push!(spvec_alloc[out_spindx][1], colval)
        push!(spvec_alloc[out_spindx][2], length(inptr_alloc[out_spindx]))

    end
    accessors = [ConnectionAccessor(
        spindx, 
        sp.colptr[spindx]:sp.colptr[spindx+1]-1, 
        sparsevec(spvec_alloc[spindx][1],spvec_alloc[spindx][2]), 
        inptr_alloc[spindx]) 
            for spindx in 1:states]
    return Accessors{eltype(g)}(accessors)
end

function setaccessors!(g)
    println("Setting")
    g.addons[:accessors] = Accessors(g.adj)
end

Base.getindex(a::Accessors, i::Integer) = a.data[i]
Base.length(a::Accessors) = length(a.data)
Base.iterate(a::Accessors, i::Int=1) = i > length(a.data) ? nothing : (a.data[i], i+1)
Base.size(a::Accessors) = length(a.data)
getnzindex(a::Accessors, i, j) = a.data[i].outidx2ptr[j]
function get_ptrs(as::Accessors, idx,jdx)
    ptridx = as[idx].outidx2ptr[jdx]
    return as[idx].outptr[ptridx], as[idx].inptr[ptridx]
end

setaccessors!(g, a::Accessors) = g.addons[:accessors] = a
function getaccessors(g::IsingGraph{T}) where T
    if !haskey(g.addons, :accessors)
        return setaccessors!(g)
    end
    return g.addons[:accessors]::Accessors{T}
end

function scale!(g, as::Accessors{etype}, idx, scale) where etype
    scale = convert(etype, scale)
    idx = Int32(idx)
    accessor = as[idx]
    outptrs = accessor.outptr
    inptrs = accessor.inptr
    adj_m = g.adj

    @turbo for idx in eachindex(outptrs)
    # for idx in eachindex(outptrs)
        ptr = outptrs[idx]
        adj_m.nzval[ptr] *= scale
    end
    @turbo for idx in eachindex(inptrs)
    # for idx in eachindex(inptrs)
        ptr = inptrs[idx]
        adj_m.nzval[ptr] *= scale
    end
end


scale!(g::IsingGraph, idx, scale) = scale!(g, getaccessors(g), idx, scale)
scale!(l::IsingLayer, idx, scale) = scale!(graph(l), idxLToG(idx, l), scale)


function set!(as::Accessors, in, out, val)
    in = Int32(in)
    out = Int32(out)
    ptrs = get_ptrs(as, in, out)
    @turbo for ptridx in 1:2
        ptr = ptrs[ptridx]
        g.adj.nzval[ptr] = val
    end
end

set!(g::IsingGraph, in, out, val) = set!(getaccessors(g), in, out, val)
set!(l::IsingLayer, in, out, val) = set!(graph(l), idxLToG(in, l), idxLToG(out, l), val)

"""
Defects to add a scaling factor to weights
"""
mutable struct SDefect{I,F}
    layer::IsingLayer
    idx::I
    scaling::F
end

getscale(sd::SDefect) = sd.scaling
getidx(sd::SDefect) = sd.idx
setidx!(sd::SDefect, idx) = sd.idx = idx
layer(sd::SDefect) = sd.layer

get_coords(sd::SDefect) = idxGToL(sd.idx, sd.layer)

function scale!(sd::SDefect, scale)
    g = graph(sd.layer)
    sd.scaling = scale
    scale!(g, idxLToG(sd.idx, sd.layer), scale)
end

# Dispatch barrier
jump!(sd::SDefect, x, y, z) = _jump!(layer(sd), sd, x, y, z)

function _jump!(l::IsingLayer, sd::SDefect, x, y, z)
    #unscale
    old_scale = getscale(sd)
    scale!(sd, 1/sd.scaling)
    (i,j,k) = idxToCoord(sd.idx, size(l))
    # Primed coordinates
    (xp, yp, zp) = coordwalk(InteractiveIsing.top(l), y+i, x+j, z+k)
    new_idx = coordToIdx(Int32.((xp, yp, zp)), size(l))
    setidx!(sd, new_idx)
    scale!(sd, old_scale)
    return sd
end

struct SDefects{I,F}
    data::Vector{SDefect{I,F}}
end

SDefects(I,F) = SDefects{I,F}(SDefect{I,F}[])

function Base.show(io::IO, sd::SDefect)
    coords = idxToCoord(sd.idx, size(sd.layer))
    println("Scaling defect at x,y,z = $((coords[2],coords[1],coords[3])) with scaling factor $(sd.scaling)")
end
Base.getindex(d::SDefects, i) = d.data[i]
Base.length(d::SDefects) = length(d.data)
Base.iterate(d::SDefects, i::Int=1) = i > length(d.data) ? nothing : (d.data[i], i+1)
Base.size(d::SDefects) = length(d.data)
Base.push!(d::SDefects, sd::SDefect) = push!(d.data, sd)

function add_sdefects!(l::IsingLayer, _scale, idxs::Integer...)
    idxs = Int32.(idxs)
    _scale = convert(eltype(graph(l)), _scale)
    for idx in idxs
        add_sdefect!(l, _scale, idx)
    end
end

function add_sdefect!(l::IsingLayer, _scale, idx::Integer)
    g = graph(l)
    T = eltype(l)
    sdefects = get_sdefects(l)
    gidx = idxLToG(idx, l)
    
    push!(sdefects, SDefect(l, Int32(idx), T(_scale)))
    scale!(g, gidx, _scale)
end



add_sdefect!(l::IsingLayer, scale, x, y, z) = add_sdefect!(l, scale, coordToIdx(Int32.((x,y,z)), size(l)))

"""
Coordinate indexing
"""
function add_sdefects!(l::IsingLayer, scale, coords::Tuple...)
    scale = convert(eltype(graph(l)), scale)
    for coord in coords
        idx = coordToIdx(Int32.(coord), size(l))
        add_sdefect!(l, idx, scale)
    end
end

function get_sdefects(g)
    ad = addons(g)
    if haskey(ad, :sdefects)
        return ad[:sdefects]::SDefects{Int32,eltype(g)}
    else
        sd = SDefects(Int32, eltype(g))
        ad[:sdefects] = sd
        return sd
    end
end