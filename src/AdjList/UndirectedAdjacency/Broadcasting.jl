@inline function _scale_storage!(vals, scalar, op)
    @inbounds for i in eachindex(vals)
        vals[i] = op(vals[i], scalar)
    end
    return vals
end

@inline function _scale_adjacency!(A::UndirectedAdjacency, scalar, op)
    _scale_storage!(A.sp.nzval, scalar, op)
    if separate_diagonal(A)
        _scale_storage!(A.diag, scalar, op)
    end
    return A
end

@inline function _try_sparse_broadcast!(A::UndirectedAdjacency, bc::Base.Broadcast.Broadcasted)
    args = bc.args
    if bc.f === (*) && length(args) == 2
        if args[1] === A && args[2] isa Number
            return _scale_adjacency!(A, args[2], *)
        elseif args[2] === A && args[1] isa Number
            return _scale_adjacency!(A, args[1], *)
        end
    elseif bc.f === (/) && length(args) == 2
        if args[1] === A && args[2] isa Number
            return _scale_adjacency!(A, args[2], /)
        end
    elseif bc.f === (-) && length(args) == 1 && args[1] === A
        return _scale_adjacency!(A, -one(eltype(A.sp.nzval)), *)
    end
    return nothing
end

function Base.copyto!(A::UndirectedAdjacency, bc::Base.Broadcast.Broadcasted)
    bc = Base.Broadcast.instantiate(bc)
    fast = _try_sparse_broadcast!(A, bc)
    fast === nothing || return fast
    return invoke(Base.copyto!, Tuple{AbstractArray, Base.Broadcast.Broadcasted}, A, bc)
end

function Base.materialize!(A::UndirectedAdjacency, bc::Base.Broadcast.Broadcasted)
    return copyto!(A, bc)
end
