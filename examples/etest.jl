import InteractiveIsing: connW, connIdx

using LoopVectorization

function getENorm(g, gstate, gadj, idx, htype)::Float32
    efac = Float32(0)
    @inbounds for conn_idx in eachindex(gadj[idx])
        conn = gadj[idx][conn_idx]
        efac += -connW(conn)*gstate[connIdx(conn)]
    end
    return efac
end

function getENormSimd(g, gstate, gadj, idx, htype)::Float32
    efac::Float32 = Float32(0)

    @inbounds @simd for conn_idx in eachindex(gadj[idx])
        conn::Tuple{Int32,Float32} = gadj[idx][conn_idx]
        @fastmath efac += -connW(conn) * gstate[connIdx(conn)]
    end

    return efac
end

function getENormSimd2(g, gstate, gadj, idx, htype)::Float32
    efactor = Float32(0)
    #= none:1 =# @inbounds #= none:1 =# @simd(for conn_idx = eachindex(gadj[idx])
                #= none:2 =#
                conn = (gadj[idx])[conn_idx]
                #= none:3 =#
                efactor += -(connW(conn)) * gstate[connIdx(conn)]
                #= none:3 =#
        end)
    return efactor
end

function getENormMag(g, gstate, gadj, idx, htype)::Float32
    efac = Float32(0)
    @inbounds for conn_idx in eachindex(gadj[idx])
        conn::Tuple{Int32,Float32} = gadj[idx][conn_idx]
        efac += -connW(conn)*gstate[connIdx(conn)] 
    end
    return efac - mlist(g)[idx]
end

getENormSum(g, gstate, gadj, idx, htype) = sum(i -> -connW(gadj[idx][i]) * gstate[connIdx(gadj[idx][i])], 1:length(gadj[idx]))

function convertAdj(adj)
newadj::Vector{Tuple{Vector{Int32},Vector{Float32}}} = []

    for idx in eachindex(adj)
        idxentry::Vector{Int32} = []
        connentry::Vector{Float32} = []
        for entryidx in eachindex(adj[idx])
            push!(idxentry, adj[idx][entryidx][1])
            push!(connentry, adj[idx][entryidx][2])
        end
      
        push!(newadj, tuple(idxentry,connentry))
    end
    return newadj
end
function convertAdj64(adj)
    newadj::Vector{Tuple{Vector{Int64},Vector{Float64}}} = []
    
        for idx in eachindex(adj)
            idxentry::Vector{Int64} = []
            connentry::Vector{Float64} = []
            for entryidx in eachindex(adj[idx])
                push!(idxentry, adj[idx][entryidx][1])
                push!(connentry, adj[idx][entryidx][2])
            end
          
            push!(newadj, tuple(idxentry,connentry))
        end
        return newadj
end

using LinearAlgebra

const newadj::Vector{Tuple{Vector{Int32},Vector{Float32}}} = convertAdj(adj(g))

function getEDot(g, gstate, gadj, idx, htype)::Float32
    idxs::Vector{Int32} = gadj[idx][1]
    weights::Vector{Float32} = gadj[idx][2]
    return @inbounds -((@view gstate[idxs]) ⋅ weights)
end

eDot(g, gstate, gadj, idx, htype)::Float32 = getEDot(g, gstate, newadj, idx, htype)::Float32


function getENormTurbo(g, gstate, gadj, idx, htype)::Float32
    efac::Float32 = Float32(0)
    idxs = newadj[idx][1]
    weights = newadj[idx][2]
    @inbounds @turbo for idx in eachindex(idxs)
        efac += -gstate[idxs[idx]]*weights[idx]
    end

    return efac
end

es(g, gstate, gadj, idx, htype) = eSNew(g, gstate, newadj, idx, htype)
function eSNew(g, gstate, gadj, idx, htype)::Float32
    efac::Float32 = Float32(0)
    idxs::Vector{Int32} = gadj[idx][1]
    weights::Vector{Float32} = gadj[idx][2]

    @inbounds @simd for conn_idx in eachindex(idxs)
        efac += - gstate[idxs[conn_idx]] * weights[conn_idx]
    end

    return efac
end