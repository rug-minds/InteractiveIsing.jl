export ParameterRef, RefContraction, SparseAdj, get_prefs, free_symb, fixed_symb, ref_symbs, is_paramref, ParameterRef, find_paramref, substitute_paramval, @ParameterRefs, ParamCollection


"""
Every AbstractParameterRef implements
free_symb : gives tuple with free indices
get_prefs : gives tuple with all refs
issparse : Says wether the ref points to a sparse structure
@generated function (::AbstractParameterRef)(freeindices) ?
expand_exp ?
"""

abstract type AbstractParameterRef end

type_apply(f, apr::Type{<:AbstractParameterRef}) = f(apr())

### PARAMTER REF
struct ParameterRef{Symb, indices, F, deref_type} <: AbstractParameterRef end

function ParameterRef(symb, indices...; func = nothing, sparse = nothing)
    F = isnothing(func) ? () : func
    deref_type = isnothing(sparse) && length(indices) == 2 ? AbstractSparseMatrix : Nothing
    return ParameterRef{Symbol(symb), (indices...,), F, deref_type}()
end

"""
Map of symbols to structs in the args
"""
const symbolmap = (;w = :(args.gadj), s = :(args.gstate), rest = :(args.params))

"""
Get the reference to the struct in either args or params
    Based on the symbol
"""
function struct_ref_exp(p::ParameterRef)
    if haskey(symbolmap, ref_symb(p))
        symbol = symbolmap[ref_symb(p)]
    else
        symbol = :($(symbolmap[:rest]).$(ref_symb(p)))
    end
end

function substruct(p::ParameterRef)
    if haskey(symbolmap, ref_symb(p))
        return nothing
    else
        return :params
    end 
end

struct_ref_exp(::Type{PR}) where PR<:AbstractParameterRef = struct_ref_exp(PR())


"""
Go from parameter ref to the struct it wants to reference
"""
@generated function get_ref(p::ParameterRef, args)
    return type_apply(struct_ref_exp, p)
end


"""
Get the indices to be filled in as expression e.g. "[i]"
"""
function  struct_ref_idx_exp(p::ParameterRef)
    return :([$(ref_indices(p)...)])
end

"""
Is ref pointing to sparse structure
"""
issparse(::ParameterRef{S, idxs, f, deref_type}) where {S, idxs, f, deref_type} = deref_type <: AbstractSparseArray

@inline function return_type(pr::ParameterRef, args)
    return eltype(get_ref(pr, args))
end


@inline function (pr::ParameterRef)(args; idxs...)
    ref = get_ref(pr, args)
    returntype = eltype(ref)
    idxs = values(idxs[free_symb(pr)])
    get_ref(pr, args)[idxs...]::returntype
end

function Base.:^(pref::ParameterRef{S, idxs, f}, power) where {S, idxs, f}
    ParameterRef{S, idxs, (f..., (^, power))}()
end

function expand_exp(pref::ParameterRef{S, idxs, f}) where {S, idxs, f}
    return Expr(:ref, struct_ref_exp(pref), free_symb(pref)...)
end

@inline function intersect_indices(pr::ParameterRef, idxs)
    idxs[free_symb(pr)]
end

### Reduce

struct RefReduce{Refs, reduce_fs} <: AbstractParameterRef end
RefReduce(refs::Tuple, reduce_fs) where N = RefReduce{refs, reduce_fs}()
free_symb(::RefReduce{Refs, reduce_fs}) where {Refs, reduce_fs} = tuple(union(free_symb.(Refs)...)...)

get_reduce_fs(::RefReduce{Refs, reduce_fs}) where {Refs, reduce_fs} = reduce_fs

Base.:+(p1::ParameterRef, p2::ParameterRef) = RefReduce((p1, p2), (+,))
Base.:-(p1::ParameterRef, p2::ParameterRef) = RefReduce((p1, p2), (-,))
Base.:+(p1::ParameterRef, p2::RefReduce) = RefReduce((p1, get_prefs(p2)...), (+, get_reduce_fs(p2)...))
Base.:-(p1::ParameterRef, p2::RefReduce) = RefReduce((p1, get_prefs(p2)...), (-, get_reduce_fs(p2)...))
Base.:+(p1::RefReduce, p2::ParameterRef) = RefReduce((get_prefs(p1)..., p2), (get_reduce_fs(p1)...,+))
Base.:-(p1::RefReduce, p2::ParameterRef) = RefReduce((get_prefs(p1)..., p2), (get_reduce_fs(p1)...,-))

issparse(::RefReduce{Refs, reduce_fs}) where {Refs, reduce_fs} = all(issparse.(Refs))

function expand_exp(rr::RefReduce{Refs, reduce_fs}) where {Refs, reduce_fs}
    return :($(Meta.parse(join(expand_exp.(Refs), string(reduce_fs)))))
end

@inline function (rr::RefReduce)(@specialize(args); idxs...)
    # @assert Set(keys(idxs)) == Set(free_symb(rr))
    refs = get_prefs(rr)
    reduce_fs = get_reduce_fs(rr)
    @inline _unroll_refreduce(gethead(refs), gettail(refs), gethead(reduce_fs), gettail(reduce_fs), args, idxs)
end

@inline function _unroll_refreduce(refshead, refstail, reduce_fshead, reduce_fstail, args, idxs)
    leftover_idxs = @inline intersect_indices(refshead, idxs)
    if isnothing(reduce_fshead)
        return @inline refshead(args; leftover_idxs...)
    end
    return reduce_fshead(refshead(args; leftover_idxs...), _unroll_refreduce(gethead(refstail), gettail(refstail), gethead(reduce_fstail), gettail(reduce_fstail), args, idxs))
end



### CONTRACTION
struct RefContraction{Refs, idxs} <: AbstractParameterRef end

function RefContraction(p1::AbstractParameterRef, p2::AbstractParameterRef)
    f1 = free_symb(p1)
    f2 = free_symb(p2)
    # refs = union(get_prefs(p1), get_prefs(p2)) 
    overlap = intersect(f1, f2)
    return RefContraction{tuple(p1,p2), tuple(overlap...)}()
end

expand_left(rc::RefContraction) = expand_exp(first(get_prefs(rc)))

Base.:*(p1::AbstractParameterRef, p2::AbstractParameterRef) = RefContraction(p1, p2)

Base.iterate(rc::RefContraction, state = 1) = iterate(get_prefs(rc), state)

# Does this one matter?
issparse(::RefContraction{Refs, idxs}) where {Refs, idxs} = all(issparse.(Refs))

# abstract type ContractionType end
# struct SparseAdj <: ContractionType end

abstract type PRefType end

abstract type VecLike <: PRefType end
abstract type MatrixLike <: PRefType end

struct VecRef <: VecLike end
struct MatrixRef <: MatrixLike end
struct SparseVecRef <: VecLike end
struct SparseMatrixRef <: MatrixLike end


abstract type ContractionType end
struct SparseColumn <: ContractionType end
struct SparseContraction <: ContractionType end
struct VectorContraction <: ContractionType end


## ACCESSORS

"""
For unified syntax to do contractions
"""
get_prefs(p::ParameterRef) = tuple(p)
get_prefs(p::RefContraction{Refs}) where Refs = Refs
get_prefs(rr::RefReduce{Refs}) where Refs = (Refs)

ref_symb(::ParameterRef{S}) where S = S
ref_symbs(::RefContraction{Refs}) where Refs = tuple(ref_symb.(Refs)...)

ref_indices(::ParameterRef{S, idxs}) where {S, idxs} = idxs
ref_indices(rr::RefReduce) = union(free_symb.(get_prefs(rr))...)
ref_indices(::Type{PR}) where PR<:AbstractParameterRef = ref_indices(PR())

free_symb(pr::ParameterRef) = ref_indices(pr)
free_symb(::RefContraction{Refs, contractions}) where {Refs, contractions} = tuple(setdiff(union(free_symb.(Refs)...), contractions))
free_symb(apr::AbstractParameterRef) = tuple(union(free_symb.(get_prefs(apr))...)...)
num_free(apr::AbstractParameterRef) = length(free_symb(apr))

contract_symb(rc::RefContraction{Rs, idxs}) where {Rs, idxs}  = idxs

## PREF FUNCTIONS
@inline function dereftype(pr::ParameterRef, args::NamedTuple)
    @inline typeof(get_ref(pr, args))
end

@inline function dereftype(pr::Union{Type{ParameterRef}, ParameterRef}, args::DataType)
    if pr isa Type
        pr = pr()
    end
    if !isnothing(substruct(pr))
        args = gettype(args, substruct(pr))
    end
    type = gettype(get_symb(pr), args)
    return type
end


"""
Get the ref type for a parameter ref
"""
function reftype(pr::AbstractParameterRef, args)
    if length(ref_indices(pr)) == 2
        if pr isa ParameterRef && dereftype(pr, args) <: AbstractSparseMatrix
            return SparseMatrixRef()
        else
            return MatrixRef()
        end
    else
        if pr isa ParameterRef && dereftype(pr, args) <: AbstractSparseVector
            return SparseVecRef()
        else
            return VecRef()
        end
    end
end

function reftype(pr::Union{Type{ParameterRef}, ParameterRef}, args::Type)
    if pr isa Type
        pr = pr()
    end
    _dereftype = dereftype(pr, args)

    if length(type_apply(ref_indices,pr)) == 2
        if pr isa ParameterRef && _dereftype <: AbstractSparseMatrix
            return SparseMatrixRef()
        else
            return MatrixRef()
        end
    else
        if pr isa ParameterRef && _dereftype <: AbstractSparseVector
            return SparseVecRef()
        else
            return VecRef()
        end
    end
end
# function reftype(pr::AbstractParameterRef, args = nothing)
#     if length(ref_indices(pr)) == 2
#         if issparse(pr)
#             return SparseMatrixRef
#         else
#             return MatrixRef
#         end
#     else
#         if issparse(pr)
#             return SparseVecRef
#         else
#             return VecRef
#         end
#     end
# end


### CONTRACTIONS

"""
Get a reference to a matrix in a contraction
    Returns nothing if it's not found
"""
function matrix_ref(rc::RefContraction, args)
    for ref in rc
        if reftype(ref, args) isa MatrixLike
            return ref
            break
        end
    end
    return nothing
end

"""
Get all vector like refs
"""
function vec_refs(rc::RefContraction, args)
    refs = get_prefs(rc)
    return tuple(_vec_refs(args, gethead(refs), gettail(refs))...)
end

function _vec_refs(args, head, tail)
    if reftype(head, args) isa VecLike
        return tuple(get_prefs(head)..., _vec_refs(args, gethead(tail), gettail(tail))...)
    else
        return tuple(_vec_refs(args, gethead(tail), gettail(tail))...)
    end
end

_vec_refs(args, ::Nothing, ::Any) = ()

vec_refs(rc::Type{PR}, args) where PR<:AbstractParameterRef = vec_refs(PR(), args)

"""
Get all references to the structs in a contraction
"""
function struct_ref_exps(ref::RefContraction)
    return struct_ref_exp.(get_prefs(ref))
end
"""
Not sure if this is needed
Get the ref to a struct
"""
function getref(args, ::Val{R}) where R
    if R == :w
        return args.gadj
    elseif R == :s
        return args.gstate
    else
        return arg.sparams.R
    end 
end

### CONTRACTION TYPES

function contraction_type(rc::RefContraction, args)
    prefs = get_prefs(rc)
    if length(prefs) == 2
        if reftype(last(prefs), args) == MatrixLike
            return SparseColumn()
        else
            return VectorContraction()
        end
    end
end

# function (rc::RefContraction)(args, params, sp_adj, fixed_indexes = 1)
#     # sp_adj = args.gadj
#     ref_symbols = ref_symbs(rc)
#     ref_symbols = filter(x -> x != :w, ref_symbols)
#     refs = getref.(Ref(args), Ref(params), Val.(ref_symbols))
#     common_zero = zero(reduce(promote_type, eltype.(refs)))
#     i = fixed_indexes
#     @turbo for ptr in nzrange(sp_adj, i)
#         j = sp_adj.rowval[ptr]
#         wij = sp_adj.nzval[ptr]
#         factor = zero(common_zero)
#         @inline reducerefs(refs, typeof(common_zero), j)
#         common_zero += wij*factor
#     end
#     return common_zero, cum
# end
"""

"""
function reducerefs(refs, type, contraction_idx)
    return type(_reducerefs(Processes.gethead(refs), Processes.gettail(refs), type, contraction_idx))
end

function _reducerefs(head, tail, type, contraction_idx)
    return type(head[contraction_idx]) + _reducerefs(Processes.gethead(tail), Processes.gettail(tail), type, contraction_idx)
end

_reducerefs(::Nothing, ::Any, ::Any, ::Any) = 0

# function ::(ParameterRef{S, idxs})(params) where {S, idxs}
#     params[S]
# end



"""
Find symbols of the form s_ij...
"""
function is_paramref(symb)
    string = String(symb)
    found = findfirst(x -> x == '_', string)
    if !isnothing(found)
        return true
    end
    return false
end

function ParameterRef(symb)
    sstr = String(symb)
    u_found = findfirst(x -> x == '_', sstr)
    get_symb = sstr[1:u_found-1]
    get_index = sstr[u_found+1:end]
    # fixed_index = tuple(Symbol.(tuple(get_index...))[1])
    # free_index = Symbol.(tuple(get_index...))[2:end]
    return ParameterRef(Symbol(get_symb), Symbol.(tuple(get_index...))...)
end

"""
Find where there are symbols of the form s_ij...
"""
function find_paramref(ex)
    symbs = []
    indexes = []
    for (idx, subex) in enumerate(ex.args)
        _find_paramref(subex, (idx,), symbs, indexes)
    end
    return symbs, indexes
end

function _find_paramref(ex, this_idxs, symbs, indexes)
    if ex isa Symbol
        if is_paramref(ex)
            push!(symbs, ex)
            push!(indexes, this_idxs)
        end
    elseif ex isa Expr
        for (this_idx, arg) in enumerate(ex.args)
            _find_paramref(arg, (this_idxs..., this_idx), symbs, indexes)
        end
    end
end

"""
Substitute in a ParameterRef
"""
function substitute_paramref(ex, indexes, symb)
    replace_symb(ex, ParameterRef(symb), indexes)
end

ParameterRefs_ex = nothing
macro ParameterRefs(ex)
    if @capture(ex, function fname_(a__) body_ end)
        symbs = find_paramref(body)
        for (symb, indexes) in zip(symbs[1], symbs[2])
            body = substitute_paramref(body, indexes, symb)
        end
        ex = quote function $fname($(a...)) $body end end
        global ParameterRefs_ex = ex
        return esc(ex)
    else
        if ex isa Symbol
            ex = Expr(:block, ex)
        end
        symbs = find_paramref(ex)
        for (symb, indexes) in zip(symbs[1], symbs[2])
            ex = substitute_paramref(ex, indexes, symb)
        end
        global ParameterRefs_ex = ex
        return esc(ex)
    end 
end

export d_ex, @ParameterRefs
ev =  @ParameterRefs function ΔiH(::Int)
    contractions = :((s_i)*w_ij)
    multiplications = :((s_i^2-sn_i^2)*self_i+(s_i-sn_i)*(b_i))
    return (;contractions, multiplications)
end

## Reductions

reduce_contraction_exp = nothing

function (rc::RefContraction{Refs})(args; idxs...) where Refs
    reduce_contraction(rc, idxs, args, reftype(first(Refs), args), reftype(last(Refs), args))
end

@generated function reduce_contraction(rc::RefContraction, idxs, args, ::VecLike, ::SparseMatrixRef)
    global reduce_contraction_exp = quote
        (;j) = idxs
        cumsum = zero(promote_eltype($(struct_ref_exp.(vec_refs(rc, args))...)))
        sp_adj = $(struct_ref_exp(type_apply(matrix_ref, rc)))
        @turbo for ptr in nzrange(sp_adj, j)
            i = sp_adj.rowval[ptr]
            wij = sp_adj.nzval[ptr]
            cumsum += wij * $(type_apply(expand_left, rc)) # Expand the left side of the contraction
                                                        # E.g. access all the refs on the left side
        end
        return cumsum
    end
    return reduce_contraction_exp
end


