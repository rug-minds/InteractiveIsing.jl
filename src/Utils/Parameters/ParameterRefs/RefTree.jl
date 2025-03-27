#### IDX TREE FOR MULTS
"""
Return ((idxs), (idxstree(first), idxstree(second), ...))
"""
function idxtree(rm::RefMult)
    connections = idxtree.(get_prefs(rm))
    idxs = getidxs.(connections)
    pure = all(ispure, connections) && all(x -> x == idxs[1], idxs)
    return ((ref_indices(rm), pure), connections)
end

idxtree(p::ParameterRef) = ((ref_indices(p), true), tuple())

getidxs(tree::Tuple) = tree[1][1]

isleaf(tree::Tuple) = isempty(tree[2])


function get_node(tree, tidxs)
    if isempty(tidxs)
        return tree
    end
    return get_node(tree[2][tidxs[1]], tidxs[2:end])
end

function nodeval(node)
    node[1]
end

nodeindices(node::Tuple) = node[1][1]
"""
A node is pure if all it's children have the same indices
    This is important because it means that the children can be grouped together
    and the operation can be performed on the whole group
"""
ispure(node::Tuple) = node[1][2] 

node_children(node) = node[2]

get_nodeval(tree, tidxs) = nodeval(get_node(tree, tidxs))

function idxtree(::Nothing)
    return tuple()
end

"""
Match the index given with the first pure node in the tree
"""
function match_indices(tree, indices, tidxs = tuple())
    this_node = get_node(tree, tidxs)
   
    if ispure(this_node) # If the node is pure indices
        if indices == nodeindices(this_node)
            return tidxs
        else
            return tuple()
        end
    else
        allin = all(map(i -> i ∈ nodeindices(this_node), indices))
        if allin
            for idx in eachindex(node_children(this_node))
                found = match_indices(tree, indices, tuple(tidxs..., idx))
                if !isempty(found)
                    return found
                end
            end
        end
    end
    return tuple()
end