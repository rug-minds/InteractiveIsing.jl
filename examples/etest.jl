import InteractiveIsing: connW, connIdx

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
        conn = gadj[idx][conn_idx]
        efac += -connW(conn) * gstate[connIdx(conn)]
    end

    return efac
end

function getENormSimd2(gstate, gadj, idx)::Float32
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
        conn = gadj[idx][conn_idx]
        efac += -connW(conn)*gstate[connIdx(conn)] 
    end
    return efac - mlist(g)[idx]
end