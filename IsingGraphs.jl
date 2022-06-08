# Ising Graph Representation and functions
# __precompile__()

# module IsingGraphs

#     export IsingGraph, hIsing, reInitGraph!, coordToIdx, idxToCoord, ising_it, paintPoint!

    mutable struct IsingGraph
        N::Int32
        size::Int32
        state::Vector{Int8}
        adj::Dict{Int32,Vector{Int32}}
        # For tracking defects
        aliveList::Vector{Int32}
        defects::Bool
        defectBools::Vector{Bool}
        defectList::Vector{Int32}
    end

    """
    INITIALIZERS
    """
        # Initialize without defects
        IsingGraph(a,b,c,d) = IsingGraph(a,b,c,d,[1:b;], false ,[false for x in 1:b],[])

        #Initialization using only N
        IsingGraph(N::Int) = IsingGraph(N,N*N,initRandomState(N),initAdj(N))

        #Initialization of graph using a state and adjacency matrix
        IsingGraph(state::Vector{Int8},adj::Dict{Int32,Vector{Int32}}) = let size = length(state)
            IsingGraph(sqrt(size), size, copy(state), adj)
        end

        # Copy graph data to new one
        IsingGraph(g::IsingGraph) = IsingGraph(g.N,g.size,copy(g.state),copy(g.adj),copy(g.aliveList),copy(g.defects),copy(g.defectBools),copy(g.defectList))

        function reInitGraph!(g::IsingGraph)
            println("Reinitializing graph")
            g.state = initRandomState(g.N)
            g.defects = false
            g.aliveList = [1:g.size;]
            g.defectBools = [false for x in 1:g.size]
            g.defectList = []
        end

    """
    Methods
    """
        # Matrix Coordinates to vector Coordinates
        @inline  function coordToIdx(i,j,N)
            return (i-1)*N+j
        end

        coordToIdx((i,j),N) = coordToIdx(i,j,N)
        # Go from idx to lattice coordinates
        @inline function idxToCoord(idx::Int,N)
            return ((idx-1)÷N+1,(idx-1)%N+1)
        end

        # Initialization of state
        function initRandomState(N)::Vector{Int8}
            return [sample([-1,1]) for x in 1:(N*N) ]
        end

        # Initialization of adjacency matrix for a given ND
        function initAdj(N)
            adj = Dict{Int32,Vector{Int32}}()
            fillAdjList!(adj,N)
            return adj
        end


        # Returns an iterator over the ising lattice
        # If there are no defects, returns whole range
        # Otherwise it returns all alive spins
        function ising_it(g::IsingGraph)
            if !g.defects
                it::UnitRange{Int32} = 1:g.size
                return it
            else
                return g.aliveList
            end

        end

    """ Setting Elements and defects """

    """ Adding Defects """

        """Helper Functions"""

            # Searches backwards from idx in list and removes item
            # This is because spin idx can only be before it's own index in aliveList
            function revRemoveSpin!(list,spin_idx)
                init = min(spin_idx, length(list)) #Initial search index
                for offset in 0:(init-1)
                    @inbounds if list[init-offset] == spin_idx
                        deleteat!(list,init-offset)
                        return init-offset # Returns index where element was found
                    end
                end
            end

            # Zip together two ordered lists into a new ordered list    
            function zipOrderedLists(vec1::Vector{T},vec2::Vector{T}) where T
                result::Vector{T} = zeros(length(vec1)+length(vec2))

                ofs1 = 1
                ofs2 = 1
                while ofs1 <= length(vec1) && ofs2 <= length(vec2)
                    @inbounds el1 = vec1[ofs1]
                    @inbounds el2 = vec2[ofs2]
                    if el1 < el2
                        @inbounds result[ofs1+ofs2-1] = el1
                        ofs1 += 1
                    else
                        @inbounds result[ofs1+ofs2-1] = el2
                        ofs2 += 1
                    end
                end
        
                if ofs1 <= length(vec1)
                    @inbounds result[ofs1+ofs2-1:end] = vec1[ofs1:end]
                else
                    @inbounds result[ofs1+ofs2-1:end] = vec2[ofs2:end]
                end
                return result
            end

            # Deletes els from vec
            # Assumes that els are in vec!
            function remOrdEls(vec::Vector{T}, els::Vector{T}) where T
                result::Vector{T} = zeros(length(vec)-length(els))
                
                it_idx = 1
                num_del = 0
                for el in els
                     while el != vec[it_idx]
                    
                        result[it_idx - num_del] = vec[it_idx]
                        it_idx +=1
                    end
                        num_del +=1
                        it_idx += 1
                end
                 result[(it_idx - num_del):end] = vec[it_idx:end]
                return result
            end

            # Remove first element equal to el and returns correpsonding index
            function removeFirst!(list,el)
                for (idx,item) in enumerate(list)
                    if item == el
                        deleteat!(list,idx)
                        return idx
                    end
                end
            end

            # Why is this slower than map?
            function testfun(spin_idxs,defectList)
                return [el for el in spin_idxs if @inbounds defectList[el] == false]
            end

        """Actually Removing and adding defects"""
            # Adding a defect to lattice
            function addDefects!(g,spin_idxs::Vector{T}) where T <: Integer
                # Only keep elements that are not defect already
                # @inbounds d_idxs::Vector{Int32} = spin_idxs[map(!,g.defectBools[spin_idxs])]
                d_idxs = spin_idxs[map(!,g.defectBools[spin_idxs])]

                if isempty(d_idxs)
                    return
                end
                
                if length(spin_idxs) > 1
                    g.defectList = zipOrderedLists(g.defectList,d_idxs)  #Add els to defectlist
                    g.aliveList = remOrdEls(g.aliveList,d_idxs) #Remove them from alivelist
                   
                else #Faster for singular elements
                    #Remove item from alive list and start searching backwards from spin_idx
                     # Since aliveList is sorted, spin_idx - idx_found gives number of smaller elements in list
                    rem_idx = revRemoveSpin!(g.aliveList,spin_idxs[1])
                    insert!(g.defectList,spin_idxs[1]-rem_idx+1,spin_idxs[1])
                end

                # Mark corresponding spins to defect
                # @inbounds g.defectBools[d_idxs] .= true
                g.defectBools[d_idxs] .= true

                # If first defect, mark lattice as containing defects
                if g.defects == false
                    g.defects = true
                end

                # Set states to zero
                # @inbounds g.state[d_idxs] .= 0
                g.state[d_idxs] .= 0

            end

            # Removing defects, insert ordered list!
            function remDefects!(g, spin_idxs::Vector{T}) where T <: Integer
              
                # Only keep ones that are actually defect
                @inbounds d_idxs = spin_idxs[g.defectBools[spin_idxs]]
                if isempty(d_idxs)
                    return
                end
          
                # Remove defects from defect list and add to aliveList
                # Assumes that els are in list!
                if length(spin_idxs) > 1
                    g.aliveList = zipOrderedLists(g.aliveList, d_idxs)
                    g.defectList = remOrdEls(g.defectList,d_idxs)
                else    #Is faster for singular elements
                    # Add to alive list
                    # Adds it to original index offset by how many smaller numbers are also removed
                    rem_idx = removeFirst!(g.defectList,spin_idxs[1]) 
                    insert!(g.aliveList,spin_idxs[1]-(rem_idx-1),spin_idxs[1])
                end

                # Spins not defect anymore
                @inbounds g.defectBools[d_idxs] .= false
         
                if isempty(g.defectList) && g.defects == true
                    g.defects = false
                end

            end

            remDefect!(g, spin_idx::T) where T <: Int = remDefects!(g,[spin_idx]) 
            addDefect!(g, spin_idx::T) where T <: Int = addDefects!(g,[spin_idx]) 
            
            # Lattice indexing
            addDefect!(g,i,j) = addDefect!(g,coordToIdx(i,j,g.N))
            remDefect!(g,i,j) = remDefect!(g,coordToIdx(i,j,g.N))

            # Add percantage of defects randomly to lattice
            function addRandomDefects!(g,p)
                if isempty(g.aliveList) || p[] == 0
                    return nothing
                end

                for def in 1:round(length(g.aliveList)*p[]/100)
                    idx = rand(g.aliveList)
                    addDefect!(g,idx)
                end
                p[] = 0         # Reset observable of percantage of elements to be poked
            end

            # Removes Multiple Defects
            function remDefects!(g,idxs::Vector{Any})
                for idx in idxs
                    remDefect!(g,idx)
                end
            end

            # Removes All defects
            function restoreState!(g)
                remDefects!(g,g.defectList)
            end

            # Restores all defects to random states 
            function restoreDefects!(g)
                nDefects = length(g.defectList) # number of defects to be restored
                states = initRandomState(nDefects) # initialize a corresponding number of random states
                @inbounds  g.state[g.defectList] .= states  # set those states to the random states
                g.aliveList = zipOrderedLists(g.aliveList,g.defectList) # Zip lists into each other, making an ordered list
                g.defectList = [] # Remove defects
            end


    """Setting Elements"""

        """Backend """
            # Set element to -1 or +1, shouldn't be 0
            function setEls!(g,spin_idxs, brush)
                # First remove defect if it was defect
                remDefects!(g,spin_idxs)
                # Then set element
                @inbounds g.state[spin_idxs] .= brush
            end

            setEl!(g,spin_idx,brush) = setEls!(g, [spin_idx], brush)
            setEl!(g,i,j,brush) =  setEl!(g,coordToIdx(i,j,g.N),brush)

        """ User Functions """
            # Set points either to element or defect
            function paintPoints!(g, coords , brush)
                idxs::Vector{Int32} = coordToIdx.(coords,g.N)
                if brush != 0
                    setEls!(g,idxs,brush)
                else
                    addDefects!(g,idxs)
                end
            end

            paintPoint!(g,i,j,brush ) = paintPoints!(g, coordToIdx(i,j,g.N) , brush)

# end