struct ProcessList{Max}
    processes::Vector{Process}
    graphidx::Vector{Int}
end
DataStructures.capacity(pl::ProcessList{M}) where M = M

ProcessList(max) = ProcessList{max}(Process[], Int[])
Base.getindex(pl::ProcessList, idx) = pl.processes[idx]

Base.length(pl::ProcessList) = length(pl.processes)
Base.iterate(pl::ProcessList, idx::Int = 1) = idx > length(pl.processes) ? nothing : (pl.processes[idx], idx + 1)
Base.empty!(pl::ProcessList) = empty!(pl.processes)


function Base.push!(pl::ProcessList, (p, idx)::Tuple{Process, Int})
    @assert length(pl.processes) < capacity(pl) "ProcessList is full" 
    push!(pl.processes, p)
    push!(pl.graphidx, idx)
end
function Base.deleteat!(pl::ProcessList, idx)
    deleteat!(pl.processes, idx)
    deleteat!(pl.graphidx, idx)
end

function processes(pl::ProcessList, gidx)
    idxs = []
    for graphidx in pl.graphidx
        if graphidx == gidx
            push!(idxs, graphidx)
        end
    end
    return @view pl.processes[idxs]
end




