using StaticArrays, SparseArrays, LoopVectorization, InteractiveIsing
struct ConnectionAccessor{N}
    spindx::Int
    outptr::UnitRange{Int64}
    inptr::SVector{N, Int32}
end

struct Accessors
    data::Vector{ConnectionAccessor}
end

function Accessors(g::IsingGraph)
    
    data = Vector{ConnectionAccessor}(undef, nstates(g))

    for spindx in 1:nstates(g)
        inptrs = Int32[]
        for (ptr, rowval) in enumerate(g.adj.rowval)
            if rowval == spindx
                push!(inptrs, ptr)
            end
        end
        data[spindx] = ConnectionAccessor(spindx, nzrange(g.adj,spindx) , SVector{length(inptrs),Int32}(inptrs...))
    end
    return Accessors(data)
end