"""
Stuff for initialization of adjacency matrix
"""
 
export fillAdjList!, numEdges, latmod, initAdj, adjToMatrix


function adjEntry!(adj, N, idx, weightFunc=defaultIsing)
    periodic = weightFunc.periodic
    NN = weightFunc.NN
    (i, j) = idxToCoord(idx, N)
    inv = weightFunc.invoke
    # Returns bool representing wether there is a connection
    # Bases connection on periodic and NN
    function doesConnect(i, j, icoup, jcoup, weight)
        # If weight is zero, does no connections
        if weight == 0.0
            return false
        end

        if periodic
            return !(icoup == i && jcoup == j)
        else
            return (!(icoup == i && jcoup == j) && (icoup > 0 && jcoup > 0 && icoup <= N && jcoup <= N))
        end
    end

    
    function coupIdxFunc(icoup, jcoup)
        if periodic
            return coordToIdx(latmod(icoup, N), latmod(jcoup, N), N)
        else
            return coordToIdx(icoup, jcoup, N)
        end
    end

    # Finds a weight for an index
    function findWeight(idx, tupls)
        for tupl in tupls
            if tupl[1] == idx
                return tupl[2]
            end
        end
    end


    entry = []
    for icoup in (i-NN):(i+NN)
        for jcoup in (j-NN):(j+NN)
            dr = sqrt((i - icoup)^2 + (j - jcoup)^2)
            weight = inv(dr, (i+icoup)/2, (j+jcoup)/2)
            if doesConnect(i, j, icoup, jcoup, weight)

                coup_idx = coupIdxFunc(icoup, jcoup)

                # Undirected graph, so copy edge weight if already present
                if idx < coup_idx
                    newWeight = weight
                else
                    newWeight = findWeight(idx, adj[coup_idx])
                end

                append!(entry, [(coup_idx, newWeight)])
            end

        end
    end
    adj[idx] = entry
end

# Should also include function!!!!
# Init the adj list of g
function fillAdjList!(adj, N, weightFunc=defaultIsing)

    for idx in 1:N*N
        adjEntry!(adj, N, idx, weightFunc)
    end

end

function adjToMatrix(adj)
    N = length(adj)
    matr = Matrix{Float32}(undef, N, N)
    for (idx, tupls) in enumerate(adj)
        for tupl in tupls
            matr[idx, tupl[1]] = tupl[2]
        end
    end
    return matr
end
