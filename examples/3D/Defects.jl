using StaticArrays, SparseArrays, LoopVectorization, InteractiveIsing
import InteractiveIsing: AverageCircular
struct ConnectionAccessor
    spindx::Int
    outptr::UnitRange{Int32}
    outidx2ptr::SparseVector
    inptr::Vector{Int32}
end

ConnectionAccessor(spindx, outptr) = ConnectionAccessor(spindx, outptr, Vector{Int32}(undef, length(outptr)), Int32[])

struct Accessors
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
    return Accessors(accessors)
end

setaccessors!(g) = g.addons[:accessors] = Accessors(g.adj)

Base.getindex(a::Accessors, i::Integer) = a.data[i]
Base.length(a::Accessors) = length(a.data)
Base.iterate(a::Accessors, i::Int=1) = i > length(a.data) ? nothing : (a.data[i], i+1)
Base.size(a::Accessors) = length(a.data)
getnzindex(a::Accessors, i, j) = a.data[i].outidx2ptr[j]

setaccessors!(g, a::Accessors) = g.addons[:accessors] = a
getaccessors(g) = g.addons[:accessors]::Accessors

function scale(g::IsingGraph, idx, scale)
    scale = convert(eltype(g), scale)
    idx = Int32(idx)
    as = getaccessors(g)
    accessor = as[idx]
    outptrs = accessor.outptr
    inptrs = accessor.inptr
    @turbo for idx in eachindex(outptrs)
        ptr = outptrs[idx]
        g.adj.nzval[ptr] *= scale
    end
    @turbo for idx in eachindex(inptrs)
        ptr = inptrs[idx]
        g.adj.nzval[ptr] *= scale
    end
end

"""
Defects to add a scaling factor to weights
"""
struct SDefect{I,F}
    idx::I
    scaling::F
end

function add_sdefect!(l::IsingLayer, idx, _scale)
    g = graph(l)
    T = eltype(g)
    get!(g.addons, :sdefects, SDefects(Int32, T))::SDefects{Int32,T}
    get!(g.addons, :accessors, Accessors(g.adj))
    gidx = idxLToG(idx, l)
    
    push!(g.addons[:sdefects], SDefect(Int32(idx), T(_scale)))
    scale(g, gidx, _scale)
end

function jump!(l, sd::SDefect, x, y, z)
    (i,j,k) = idxGToL(sd.idx, l)

end


struct SDefects{I,F}
    data::Vector{SDefect{I,F}}
end

SDefects(I,F) = SDefects{I,F}(SDefect{I,F}[])

Base.getindex(d::SDefects, i::Int) = d.data[i]
Base.length(d::SDefects) = length(d.data)
Base.iterate(d::SDefects, i::Int=1) = i > length(d.data) ? nothing : (d.data[i], i+1)
Base.size(d::SDefects) = length(d.data)
Base.push!(d::SDefects, sd::SDefect) = push!(d.data, sd)


function add_sdefects!(l::IsingLayer, scale, coords::Tuple...)
    for coord in coords
        idx = coordToIdx(Int32.(coord), size(l))
        add_sdefect!(l, idx, scale)
    end
end

getdefects(g) = g.addons[:sdefects]::SDefects{Int32,eltype(g)} 
