export FlipProposal, to_delta_exp, to_delta_exp!

"""
rule to replace parameter :symb[i] with :((@view symb[]))
"""
struct FlipProposal{symb, index_symb, T} <: AbstractProposal
    at_idx::Int
    from_val::T
    to_val::T
    layer_idx::Int
    accepted::Bool
end

function FlipProposal(fp::FlipProposal{S, Idx, T}, at_idx, from_val, to_val, layer_idx, accept = false) where {S, Idx, T}
    return FlipProposal{S, Idx, T}(at_idx, from_val, to_val, layer_idx, accept)
end

function delta(fp::FlipProposal{S, Idx, T}) where {S, Idx, T}
    return fp.to_val - fp.from_val
end

function accept(state, f::FlipProposal)
    state[at_idx(f)] = to_val(f)
    FlipProposal(f, f.at_idx, f.from_val, f.to_val, f.layer_idx, true)
end
isaccepted(r::FlipProposal) = r.accepted

at_idx(r::FlipProposal) = r.at_idx
from_val(r::FlipProposal) = r.from_val
to_val(r::FlipProposal) = r.to_val

# function FlipProposal(s::Symbol; kwargs...) 
#     @assert length(kwargs) == 1 "FlipProposal requires exactly one index => value pair"
#     for (k,p) in kwargs
#         @assert p isa Pair{Int, T} where T "FlipProposal index-value pair must be of the form Int => T"
#         return FlipProposal{s, k, typeof(p[2])}(p[1], p[2])
#     end
# end

Base.size(r::FlipProposal) = (1,)
Base.length(r::FlipProposal) = 1
Base.eltype(r::FlipProposal{S, Idx, T}) where {S, Idx, T} = T

SparseArrays.rowvals(r::FlipProposal) = [r.at_idx]
SparseArrays.nonzeros(r::FlipProposal) = [r.to_val]
# SparseArrays.isassigned(r::FlipProposal) = true

getsymb(r::Type{FlipProposal{S, Idx, T}}) where {S, Idx, T} = S
getsymb(r::FlipProposal) = getsymb(typeof(r))
getindex_symb(r::Type{FlipProposal{S, Idx, T}}) where {S, Idx, T} = Idx
getindex_symb(r::FlipProposal) = getindex_symb(typeof(r))

@inline function Base.getindex(r::FlipProposal, i::Int = r.at_idx)
    i == r.at_idx ? r.to_val : eltype(r)(0)
end

Base.setindex!(r::FlipProposal, v, at_idx::Int) = begin r.at_idx = at_idx; r.to_val = v end

getidx(r::FlipProposal) = r.at_idx
getval(r::FlipProposal) = r.to_val
"""
Replaces symbols in the expression with the form symb[idx_symb] with (symb[idx_symb]- delta_i[])
    where i is the index of the FlipProposal in the FlipProposals list
"""
function to_delta_exp(expr, deltas::Union{FlipProposal, Type{<:FlipProposal}}...)
    # Kwargs are given as s = ()
    return MacroTools.postwalk(x -> begin
        for (didx, d) in enumerate(deltas)
            s = getsymb(d)
            si = getindex_symb(d)

            if @capture(x, symb_[idx_])
                    # println("Matched: ", x, " with symb: ", symb, " idx: ", idx)
                    if s == symb && si == idx
                        # println("Replacing ", x, " with delta view")
                        x = :($(Symbol("delta_", didx))[$si] - $symb[$si])
                    end
            end
        end
        # println("Returning x: ", x)
        return x
    end, expr)
end

@inline @generated function to_delta_exp_static(expr, deltas::Union{FlipProposal, Type{<:FlipProposal}}...)
    # Kwargs are given as s = ()
    return MacroTools.postwalk(x -> begin
        for (didx, d) in enumerate(deltas)
            s = getsymb(d)
            si = getindex_symb(d)

            if @capture(x, symb_[idx_])
                    # println("Matched: ", x, " with symb: ", symb, " idx: ", idx)
                    if s == symb && si == idx
                        # println("Replacing ", x, " with delta view")
                        x = :($(Symbol("delta_", didx))[$si]- $symb[$si])
                    end
            end
        end
        # println("Returning x: ", x)
        return x
    end, expr)
end

"""
From a set of FlipProposals, get the indexes that are fixed by them
TODO: This doesn't need to find unique ones
In fact it needs to @assert that they are unique
"""
function FlipProposals_to_fixed_idxs(FlipProposals::Union{FlipProposal, Type{<:FlipProposal}}...)
    fixed_idxs = []
    d_idxs = []
    for (i, d) in enumerate(FlipProposals)
        if getindex_symb(d) in fixed_idxs
            continue
        end
        push!(fixed_idxs, getindex_symb(d))
        push!(d_idxs, i)
    end
    return getindex_symb.(FlipProposals[d_idxs]), d_idxs
end

function wrap_in_call(exprs, func::Symbol, args...)
    return Expr.(Ref(:call), Ref(func), exprs, Ref.(args)...)
end