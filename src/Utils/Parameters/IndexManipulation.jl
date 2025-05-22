"""
If indexes are given as i=j, replace the j's with i
"""
function replace_indices(indexset, replacerule)

end

function index_to_axis(indices)
    (;ntuple(i -> indices[i] => i, length(indices))...)
end

"""
Find the first ref from a set of refs that includes the index
"""
function index_first_ref(refs, index)
    indices = ref_indices.(refs)
    for idxs in indices
        i = findfirst(x -> x == index, idxs)
        if i !== nothing
            return i
        end
    end
    return nothing
end

"""
Find which indexes are contracting based on the indexes that are given
"""
function idx_subtract(ref_ind, fill_ind::NamedTuple)
    if !(isempty(fill_ind))
        return tuple(setdiff(ref_ind, fill_ind)...)
    else
        return ref_ind
    end
end

function idx_subtract(ref_ind, fill_ind::Tuple)
    if !(isempty(fill_ind))
        return tuple(setdiff(ref_ind, fill_ind)...)
    else
        return ref_ind
    end
end


