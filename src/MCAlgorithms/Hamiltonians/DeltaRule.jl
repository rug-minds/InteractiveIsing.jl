export DeltaRule, to_delta_exp, to_delta_exp!

"""
rule to replace parameter :symb[i] with :((@view symb[]))
"""
mutable struct DeltaRule{symb, index_symb, T} <: AbstractSparseArray{Int, T,1}
    idx::Int
    val::T
end

function DeltaRule(s::Symbol; kwargs...) 
    @assert length(kwargs) == 1 "DeltaRule requires exactly one index => value pair"
    for (k,p) in kwargs
        @assert p isa Pair{Int, T} where T "DeltaRule index-value pair must be of the form Int => T"
        return DeltaRule{s, k, typeof(p[2])}(p[1], p[2])
    end
end

Base.size(r::DeltaRule) = (1,)
Base.length(r::DeltaRule) = 1
Base.eltype(r::DeltaRule{S, Idx, T}) where {S, Idx, T} = T

SparseArrays.rowvals(r::DeltaRule) = [r.idx]
SparseArrays.nonzeros(r::DeltaRule) = [r.val]
# SparseArrays.isassigned(r::DeltaRule) = true

getsymb(r::Type{DeltaRule{S, Idx, T}}) where {S, Idx, T} = S
getsymb(r::DeltaRule) = getsymb(typeof(r))
getindex_symb(r::Type{DeltaRule{S, Idx, T}}) where {S, Idx, T} = Idx
getindex_symb(r::DeltaRule) = getindex_symb(typeof(r))

function Base.getindex(r::DeltaRule, i::Int = r.idx)
    i == r.idx ? r.val : eltype(r)(0)
end

Base.setindex!(r::DeltaRule, v, idx::Int) = begin r.idx = idx; r.val = v end


getidx(r::DeltaRule) = r.idx
getval(r::DeltaRule) = r.val

"""
Replaces symbols in the expression with the form symb[idx_symb] with (symb[idx_symb]- delta_i[])
    where i is the index of the deltarule in the deltarules list
"""
function to_delta_exp(expr, deltas::Union{DeltaRule, Type{<:DeltaRule}}...)
    # Kwargs are given as s = ()
    return MacroTools.postwalk(x -> begin
        for (didx, d) in enumerate(deltas)
            s = getsymb(d)
            si = getindex_symb(d)

            if @capture(x, symb_[idx_])
                    # println("Matched: ", x, " with symb: ", symb, " idx: ", idx)
                    if s == symb && si == idx
                        # println("Replacing ", x, " with delta view")
                        x = :($symb[$si]-$(Symbol("delta_", didx))[$si])
                    end
            end
        end
        # println("Returning x: ", x)
        return x
    end, expr)
end

"""
From a set of DeltaRules, get the indexes that are fixed by them
TODO: This doesn't need to find unique ones
In fact it needs to @assert that they are unique
"""
function deltarules_to_fixed_idxs(deltarules::Union{DeltaRule, Type{<:DeltaRule}}...)
    fixed_idxs = []
    d_idxs = []
    for (i, d) in enumerate(deltarules)
        if getindex_symb(d) in fixed_idxs
            continue
        end
        push!(fixed_idxs, getindex_symb(d))
        push!(d_idxs, i)
    end
    return getindex_symb.(deltarules[d_idxs]), d_idxs
end

function wrap_in_call(exprs, func::Symbol, args...)
    return Expr.(Ref(:call), Ref(func), exprs, Ref.(args)...)
end


const ΔH_expr = Dict{Type, Expr}()
get_ΔH_expr(::Type{H}) where {H} = ΔH_expr[Base.typename(H).wrapper]
get_ΔH_expr(h::H) where {H} = get_ΔH_expr(H)
gen_ΔH_expr = nothing
generated_func_calls = nothing