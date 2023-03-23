import InteractiveIsing: connW, connIdx

using LoopVectorization, SIMD

function getENorm(g, gstate, gadj, idx, htype = htype(g))::Float32
    efac = Float32(0)
    @inbounds for conn_idx in eachindex(gadj[idx])
        conn = gadj[idx][conn_idx]
        efac += -connW(conn)*gstate[connIdx(conn)]
    end
    return efac
end

function getENormSimd(g, gstate, gadj, idx, htype = htype(g))::Float32
    efac::Float32 = Float32(0)

    @inbounds @simd for conn_idx in eachindex(gadj[idx])
        conn::Tuple{Int32,Float32} = gadj[idx][conn_idx]
        efac += -connW(conn) * gstate[connIdx(conn)]
    end

    return efac
end

function getENormSimd2(g, gstate, gadj, sidx, htype = htype(g))::Float32
    efac::Float32 = Float32(0)
    entry::Vector{Conn} = gadj[sidx]

    @inbounds @simd for conn_idx in eachindex(gadj[sidx])
        conn::Conn = entry[conn_idx]
        efac += -weight(conn) * gstate[idx(conn)]
    end

    return efac
end

struct Conn
    idx::Int32
    weight::Float32
end
@inline idx(c::Conn)::Int32 = c.idx
@inline weight(c::Conn)::Float32 = c.weight

function adjOfEntry(adj)
    newadj::Vector{Vector{Conn}} = []
    for conn_idx in eachindex(adj)
        entries = []
        for entry in adj[conn_idx]
            push!(entries, Conn(entry[1],entry[2]))
        end
        push!(newadj, entries)
    end
    return newadj
end

function getETurbo(g, gstate, gadj, idx, htype = htype(g))::Float32
    conns::Vector{Conn} = gadj[idx]

    len = length(conns)
    efac = Float32(0)

    @turbo for c_idx in 1:len
        wt = getfield( conns[c_idx], :weight)
    
        efac += wt
    end
    
    return efac

end

function turboman1(tuples, gstate)
    efac = Float32(0)

    @turbo for idx in eachindex(tuples)
        efac += -tuples[idx][2] * gstate[tuples[idx][1]]
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

getENormSum(g, gstate, gadj, idx, htype) =let adjentry = gadj[idx]; sum(i -> -connW(adjentry[i]) * gstate[connIdx(adjentry[i])], 1:length(gadj[idx])) end

struct AdjEntry
    idxs::Vector{Int32}
    weights::Vector{Float32}
end

@inline AdjEntry() = AdjEntry([],[])

@inline idxs(ae::AdjEntry) = ae.idxs
@inline weights(ae::AdjEntry) = ae.weights


#### Functions with different adj representation ###
function convertAdj(adj)
    newadj::Vector{AdjEntry} = []

    for idx in eachindex(adj)
        entry = AdjEntry()
        # idxentry::Vector{Int32} = []
        # connentry::Vector{Float32} = []
        for entryidx in eachindex(adj[idx])
            push!(idxs(entry), adj[idx][entryidx][1])
            push!(weights(entry), adj[idx][entryidx][2])
        end
      
        push!(newadj, entry)
    end
    return newadj
end

# function convertAdj64(adj)
#     newadj::Vector{Tuple{Vector{Int64},Vector{Float64}}} = []
    
#         for idx in eachindex(adj)
#             idxentry::Vector{Int64} = []
#             connentry::Vector{Float64} = []
#             for entryidx in eachindex(adj[idx])
#                 push!(idxentry, adj[idx][entryidx][1])
#                 push!(connentry, adj[idx][entryidx][2])
#             end
          
#             push!(newadj, tuple(idxentry,connentry))
#         end
#         return newadj
# end

using LinearAlgebra

const newadj::Vector{AdjEntry} = convertAdj(adj(g))

function getEDot(g, gstate, gadj, idx, htype)::Float32
    idxvec::Vector{Int32} = idxs(gadj[idx])
    weights::Vector{Float32} = weights(gadj[idx])
    return @inbounds -((@view gstate[idxvec]) ⋅ weights)
end

eDot(g, gstate, gadj, idx, htype)::Float32 = getEDot(g, gstate, newadj, idx, htype)::Float32

function eSNew(g, gstate, gadj, idx, htype)::Float32
    efac::Float32 = Float32(0)
    idxs::Vector{Int32} = idxs(gadj[idx])
    weights::Vector{Float32} = weights(gadj[idx])

    @inbounds for conn_idx in eachindex(idxs)
        efac += - gstate[idxs[conn_idx]] * weights[conn_idx]
    end

    return efac
end

function getETurboNew(g, gstate, gadj, idx, htype = htype(g))::Float32
    efac::Float32 = Float32(0)
    # @inbounds idxsv::Vector{Int32} = idxs(gadj[idx])
    # @inbounds weightsv::Vector{Float32} = weights(gadj[idx])

    @turbo for idx in eachindex(gadj[idx].idxs)
        efac += -gstate[(gadj[idx].idxs)[idx]]*(gadj[idx].weights)[idx]
    end

    return efac
end

function newTurbo(g, gstate, gadj, idx, htype)::Float32
    @inbounds idxsv::Vector{Int32} = idxs(gadj[idx])
    @inbounds weightsv::Vector{Float32} = weights(gadj[idx])
    # stateview = view(gstate, idxs)
    stateview = view(gstate,idxsv)
    return turboman(stateview, weightsv)
end

function turboman(stateview,weightsv)::Float32
    efac = Float32(0)

    @turbo for i in eachindex(stateview)
        efac += -stateview[i]*weightsv[i]
    end
    return efac
end

### State Rep Float32 ###

const state32 = Float32.(state(g))
const adj32 = convertAdj(adj(g))

eSumView(g, gstate, gadj, idx, htype) = let stateview = view(gstate, gadj[idx][1]), weights = gadj[idx][2]; sum(i -> ( - weights[i] * stateview[i]), 1:length(gadj[idx][1])) end

function manESimd(g, stateview, gadj, idx, htype = htype(g) )::Float32
    efac::Float32 = Float32(0)    
    @inbounds idxs::Vector{Int32} = idxs(gadj[idx])
    @inbounds weights::Vector{Float32} = weights(gadj[idx])

    lane = VecRange{4}(0)
    # @inbounds for idx in eachindex(idxs)
    @inbounds for idx in 1:4:length(idxs)
        efac += - sum(weights[lane+idx]*stateview[lane+idx])
        # efac += sum(stateview[lane+idx])
    end

    return efac
end

# @btime getENorm(g, state32, adj(g), 1, htype(g))
# @btime getENorm(g, state(g), adj(g), 1, htype(g))


# @btime getENormSimd(g, state32, adj(g), 1, htype(g))
# @btime getENormSimd(g, state(g), adj(g), 1, htype(g))

# @btime getETurbo(g, state32, adj(g), 1, htype(g))


# @btime es($g, state32, adj32, 1, htype(g))
# @btime es($g, state(g), adj32, 1, htype(g))

@benchmark getETurboNew(g, state32, adj32, 1, htype(g))
@benchmark getENormSimd(g, state(g), adj(g), 1)
@benchmark newTurbo(g, state32, adj32, 1, htype(g))

