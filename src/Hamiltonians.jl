# Hamiltonians

struct HType{Weighted,Magfield,Clamp} end

# No weights
function HFunc(g::AbstractIsingGraph,idx)::Float32
    
    efactor::Float32 = 0.
    for conn in g.adj[idx]
        @inbounds efactor += -g.state[connIdx(conn)]
    end

    return efactor
end

# No weights but magfield
function HMagFunc(g::AbstractIsingGraph,idx)::Float32
    
    efactor::Float32 = 0.
    for conn in g.adj[idx]
        @inbounds efactor += -g.state[connIdx(conn)]
    end

    return efactor -g.d.mlist[idx]
end

# When there's weights
function HWeightedFunc(g::AbstractIsingGraph,idx)::Float32
    efactor::Float32 = 0.
    for conn in g.adj[idx]
        @inbounds efactor += -connW(conn)*g.state[connIdx(conn)]
    end
    return efactor
end

# Weights and magfield
function HWMagFunc(g::AbstractIsingGraph,idx)::Float32
    efactor::Float32 = 0.
    for conn in g.adj[idx]
        @inbounds efactor += -connW(conn)*g.state[connIdx(conn)]
    end
    return efactor -g.d.mlist[idx]
end



# Sets magnetic field and branches simulation
# Prints function set if prt
function setGHFunc!(sim, prt = true)
    g = sim.g
    if !g.d.weighted
        if !g.d.mactive
            g.d.hFuncRef = Ref(HFunc)
            if prt
                println("Set HFunc")
            end
        else
            g.d.hFuncRef = Ref(HMagFunc)
            if prt
                println("Set HMagFunc")
            end
        end
    else
        if !g.d.mactive
            g.d.hFuncRef = Ref(HWeightedFunc)
            if prt
                println("Set HWeightedFunc")
            end
        else
            g.d.hFuncRef = Ref(HWMagFunc)
            if prt
                println("Set HWMagFunc")
            end
        end
    end

    branchSim(sim)
end

# Hamiltonian elements
function unweightedFac(g::AbstractIsingGraph,idx, state = state[idx])::Float32
    efactor::Float32 = 0.

    for _ in g.adj[idx]
        @inbounds efactor += -state
    end

    return efactor
end

function weightedFac(g::AbstractIsingGraph,idx, state = state[idx])::Float32
    efactor::Float32 = 0.
    for conn in g.adj[idx]
        @inbounds efactor += -connW(conn)*state
    end
    return efactor
end

function magFac(g::AbstractIsingGraph,idx)::Float32
    return - g.d.mlist[idx]
end

export getHFac
function getHFac(sim, prt)
    g = sim.g

    expr = Expr(:block,:( (g::AbstractIsingGraph,idx, state = state[idx])::Float32 ),:(->) )
    
    if g.d.weighted
        fac_exprs = [:(unweightedFac(g,idx,state))]
    else
        fac_exprs = [:(weightedFac(g,idx,state))]
    end
    
    if g.d.weighted
        append!(fac_exprs, [:(+ magFac(g, idx))] )
    end
    return Expr(:block, expr, fac_exprs...) 
end


# @generated genHamiltonian(htype::HType)
# 
# end