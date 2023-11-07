include("LayeredMetropolis.jl")
include("LayeredLangevin.jl")

"""
From the type of the named tuple containing the layers and the graphidxs 
of the layers, get the layer types and the indexes of the layers
"""
function get_layertypes_idxs(argstype)
    layertypes = gettype(argstype, :layers).parameters
    ## TODO: MAYBE MOVE THIS INTO THE LAYER TYPE
    layeridxs = getval(gettype(argstype, :l_iterators))
    return layertypes, layeridxs
end

"""
Group the layer types and graphidxs of the types
"""
function grouped_ltypes_idxs(argstype)
    layertypes, layeridxs = get_layertypes_idxs(argstype)
    grouped_ltypes = []
    grouped_idxs = []
    current_ltype = layertypes[1]
    current_idxs = layeridxs[1]
    for idx in eachindex(layertypes)
        if layertypes[idx] == current_ltype # Then group indexes
            # Iterator for grouped layers is begin index of first item and end index of last item
            current_idxs = typeof(current_idxs)(current_idxs[1]:layeridxs[idx][end])
        else # Push the group and start a new one
            push!(grouped_ltypes, current_ltype)
            push!(grouped_idxs, current_idxs)
            current_ltype = layertypes[idx]
            current_idxs = layeridxs[idx]
        end
    end
    # If ended, push the last group
    push!(grouped_ltypes, current_ltype)
    push!(grouped_idxs, current_idxs)
    return zip(grouped_ltypes, grouped_idxs)
end