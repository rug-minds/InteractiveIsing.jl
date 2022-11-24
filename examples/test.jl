function getEFactor1(g,idx)
    esum = 0
    list = g.adj[1]
    for conn in list
            esum += connW(conn)*g.state[connIdx(conn)]
    end
    return esum
end

function getEFactor2(g,idx)
    list = g.adj[idx]
    return -sum(connW.(list).* @inbounds (@view g.state[connIdx.(list)]))
end

function getEFactor3()
    esum = 0
    list = g.adj[idx]
    for conn in list
            esum += connW(conn)*g.state[connIdx(conn)]
    end
    return esum
end

function getEFactor4()
    list = g.adj[idx]
    return -sum(connW.(list).* @inbounds (@view g.state[connIdx.(list)]))
end

function adjToSparse(adj)
    matr = spzeros(500^2,500^2)
    for (idx,row) in enumerate(adj)
        for conn in row
            matr[idx,connIdx(conn)] = connW(conn)
        end
    end
    return matr
end

function getEFactorM(g,row, matr)
    esum = 0
    for (idx,weight) in enumerate(@view matr[row,:])
        esum += -weight*g.state[idx]
    end
end