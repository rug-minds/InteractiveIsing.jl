#=
Stuff for initialization of adjacency matrix
=#
 
export fillAdjList!, numEdges, latmod, adjToMatrix

"""
Creates an entry for an adjacency list
I.e. a vector containing outgoing edges for a given spin will be created
"""
function adjEntry(adj, length, width, idx, periodic, NN, inv)
    # Returns bool representing wether there is a connection
    # Bases connection on periodic and NN
    function doesConnect(i, j, icoup, jcoup, weight)
        # If weight is zero, does not connect
        ! Bool(weight) && return false

        # Filter out self connections and, if not periodic, connections that are out of the grid
        if periodic
            return !(icoup == i && jcoup == j)
        else
            return (!(icoup == i && jcoup == j) && (icoup > 0 && jcoup > 0 && icoup <= length && jcoup <= width))
        end
    end

    
    # If periodic, loop coordinates arround
    function coupIdxFunc(icoup, jcoup)
        if periodic
            return coordToIdx(latmod(icoup, length), latmod(jcoup, width), length)
        else
            return coordToIdx(icoup, jcoup, length)
        end
    end

    # Finds a weight for an index
    function findWeight(idx, tupls)
        for tupl in tupls
            if tupl[1] == idx
                return tupl[2]
            end
        end
        # If not found, should fail
        return nothing
    end
 
    # Coordinates of idx
    (i, j) = idxToCoord(idx, length)
    # Entry of adj
    entry = []
    for jcoup in (j-NN):(j+NN)
        for icoup in (i-NN):(i+NN)
            dr = sqrt((i - icoup)^2 + (j - jcoup)^2)
            # Generate weight based on relative dist
            weight = inv(dr, (i+icoup)/2, (j+jcoup)/2)

            # If not same point or out of lattice if not periodic
            if doesConnect(i, j, icoup, jcoup, weight)
                
                # Idx of other spin
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
    return sort!(entry)
end

""" 
Initialize an adjacencylist to be used in an IsingGraph
"""
function fillAdjList!(adj, length, width, weightFunc=defaultIsingWF)
    periodic = weightFunc.periodic
    NN = weightFunc.NN
    inv = weightFunc.invoke

    for idx in 1:length*width
        adj[idx] = adjEntry(adj, length, width, idx, periodic, NN, inv)
    end
end

"""
Reads an adjacency list as a matrix
"""
function adjToMatrix(adj, length, width)
    matr = Matrix{Float32}(undef, length, width)
    for (idx, tupls) in enumerate(adj)
        for tupl in tupls
            matr[idx, tupl[1]] = tupl[2]
        end
    end
    return matr
end

adjToMatrix(g) = adjToMatrix(adj(g), glength(g), gwidth(g))


function setAdj!(sim, layer, wf)
    g = sim.layers[layer]
    adj(g) = initSqAdj(glength(g), gwidth(g), weightFunc = wf)
    setSimHType!(sim, :NN => wf.NN)
end
export setAdj!
