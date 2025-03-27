export ParameterRef, RefMult, SparseAdj, get_prefs, free_symb, fixed_symb, ref_symbs, is_paramref, ParameterRef, find_paramref, substitute_paramval, @ParameterRefs, ParamCollection

include("Utils.jl")

abstract type FType end
struct IdentityF <: FType end
struct Unary <: FType end
struct Binary <: FType end

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
struct ParameterRef{Symb, indices, F, D} <: AbstractParameterRef 
    data::D
end

"""
Indices in this node stay the same for every nody downwards in the tree
"""
ispure(::ParameterRef) = true
ispure(::Type{PR}) where PR<:AbstractParameterRef = ispure(PR())
ispureval(::ParameterRef) = Val(true)
ispureval(::Type{PR}) where PR<:AbstractParameterRef = Val(ispure(PR()))

get_prefs(p::ParameterRef) = tuple(p)

ref_symb(::ParameterRef{S}) where S = S #Get the symbol of the reference
ref_symb(::Type{PR}) where PR<:AbstractParameterRef = ref_symb(PR())
ref_indices(::ParameterRef{S, idxs}) where {S, idxs} = idxs #Get the indices
"""
Get a set of the indices present in the type of an AbstractParameterRef
"""
ref_indices(::Type{PR}) where PR<:AbstractParameterRef = ref_indices(PR())

get_F(::ParameterRef{S, idxs, F}) where {S, idxs, F} = F
get_F(::Type{PR}) where PR<:AbstractParameterRef = get_F(PR())

function expr_F_wrap(apr::AbstractParameterRef, expr)
    if isempty(get_F(apr))
        return expr
    elseif F_type(apr) isa Unary
        return Expr(:call, get_F(apr)[1], expr)
    elseif F_type(apr) isa Binary
        return Expr(:call, get_F(apr)[1], expr, get_F(apr)[2])
    end
end

function apply_F(apr::AbstractParameterRef, result)
    f = get_F(apr)
    if isempty(f)
        return result
    elseif F_type(apr) isa Unary
        return f[1](result)
    elseif F_type(apr) isa Binary
        return f[1](result, f[2])
    end
end


function ParameterRef(symb, indices...; func = nothing, data = nothing)
    F = isnothing(func) ? () : func
    return ParameterRef{Symbol(symb), (indices...,), F, typeof(data)}(data)
end

"""
Type forward for generated
"""
ParameterRef{A,B,C,D}() where {A,B,C,D} = ParameterRef{A,B,C,Nothing}(nothing)


function Base.getindex(pr::AbstractParameterRef, idx)
    if pr isa ParameterRef
        error("Cannot index a ParameterRef")
    end
    get_prefs(pr)[idx]
end


function refmap(::Val{:w})
    return (:gadj,)
end

function refmap(::Val{:self})
    return (:g, :self)
end

function refmap(::Val{:s})
    return (:gstate,)
end

function refmap(::Val{:sn})
    return (:newstate,)
end

function refmap(::Val{A}) where A
    return (:hamiltonian,:($A))
end

"""
Get the reference to the struct in either args or params
    Based on the symbol
"""
function struct_ref_exp(p::ParameterRef)
    path = tuple(refmap(Val(ref_symb(p)))...)
    return (build_getproperty_chain(:args, path),)
end

function substructs(p::ParameterRef)
    if length(struct_ref_exp(p)) == 1
        return nothing
    else
        return struct_ref_exp(p)[1:end-1]
    end
end

struct_ref_exp(::Type{PR}) where PR<:AbstractParameterRef = struct_ref_exp(PR())

build_getproperty_chain(symbs::Tuple) = build_getfield_chain(first(symbs), Base.tail(symbs))
function build_getproperty_chain(base_expr, symbols::Tuple)
    if isempty(symbols)
        return base_expr
    end
    return build_getproperty_chain(:(getproperty($base_expr, $(QuoteNode(first(symbols))))), Base.tail(symbols))
end



@generated function get_ref(p, args::DataType)
    return :(args.$(ref_symb(p)))
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
issparse(p::ParameterRef{S, idxs, f}, args) where {S, idxs, f} = dereftype(p, args) <: AbstractSparseArray

@inline function return_type(pr::ParameterRef, args)
    return eltype(get_ref(pr, args))
end

@inline function return_type(pt::NTuple{N, <:AbstractParameterRef}, args) where N
    return promote_type(return_type.(pt, Ref(args))...)
end


paramref_type_exp = nothing
paramref_type_args = []
@inline @generated function (pr::ParameterRef)(args; idxs...)
    # ref = get_ref(pr, args)
    # returntype = eltype(ref)
    # idxs = values(idxs[free_symb(pr)])
    # get_ref(pr, args)[idxs...]::returntype
    ref_exp = struct_ref_exp(pr())
    ri = ref_indices(pr())
    exp = Expr(:call, :getindex, ref_exp..., ri...)
    global paramref_type_args = [pr, args, idxs]
    global paramref_type_exp = quote
        $(ri...) = idxs[$(QuoteNode(ri...))]
        returntype = return_type(pr, args)
        return ($exp)::returntype
    end
    return paramref_type_exp
end

function Base.:^(pref::ParameterRef{S, idxs, f, d}, power) where {S, idxs, f, d}
    # ParameterRef{S, idxs, (f..., (^, power))}()
    # ParameterRef(symb, indices...; func = nothing, data = nothing)
    ParameterRef(ref_symb(pref), ref_indices(pref)...; func = (^, power), data = pref.data)
end

expand_exp(pr::Type{<:AbstractParameterRef}) = expand_exp(pr())
function expand_exp(pref::ParameterRef{S, idxs, f}) where {S, idxs, f}
    if isempty(f)
        return Expr(:ref, struct_ref_exp(pref)..., free_symb(pref)...)
    else
        return Expr(:call, f[1], Expr(:ref, struct_ref_exp(pref)..., free_symb(pref)...), f[2])
    end
end

@inline function intersect_indices(pr::AbstractParameterRef, idxs)
    idxs[free_symb(pr)]
end


include("RefReduce.jl")
include("RefMult.jl")
include("RefTree.jl")


# abstract type ContractionType end
# struct SparseAdj <: ContractionType end

abstract type PRefType end

abstract type VecLike <: PRefType end
abstract type MatrixLike <: PRefType end

struct VecRef <: VecLike end
struct MatrixRef <: MatrixLike end

sparsify(::VecRef, bool) = bool ? SparseVecRef() : VecRef()
sparsify(::MatrixRef, bool) = bool ? SparseMatrixRef() : MatrixRef()

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


@generated function ref_indices_val(ar::AbstractParameterRef)
    t = Val.(ref_indices(ar()))
    return :($t)
end

free_symb(pr::ParameterRef) = ref_indices(pr)
@generated function free_symb(::RefMult{Refs, contractions}) where {Refs, contractions}
    t = tuple(setdiff(union(free_symb.(Refs)...), contractions))
    return :($t)
end
@generated function free_symb(apr::AbstractParameterRef)
    t = tuple(union(free_symb.(get_prefs(apr))...)...)
    return :($t)
end

num_free(apr::AbstractParameterRef) = length(free_symb(apr))



"""
Go from parameter ref to the struct it wants to reference
"""
# @generated function get_ref(p::ParameterRef, args)
#     return type_apply(struct_ref_exp, p)
# end
get_ref_exp = nothing
@generated function get_ref(p, args::Union{<:NamedTuple, <:Base.Pairs})
    symb = type_apply(ref_symb, p)
    refs = refmap(Val(symb))
    global get_ref_exp = :($(build_getproperty_chain(:args, refs)))
    return get_ref_exp
end

## PREF FUNCTIONS
@inline function dereftype(pr::ParameterRef, args::NamedTuple)
    @inline typeof(get_ref(pr, args))
end

@inline function dereftype(pr, args::DataType)
    struct_refs = refmap(Val(ref_symb(pr)))
    return gettype(args, struct_refs)
end


"""
Get the ref type for a parameter ref
"""
function reftype(pr, args)
    apriori_type = length(ref_indices(pr)) > 1 ? MatrixRef() : VecRef()
    
    if !(pr isa ParameterRef)
        return apriori_type
    end

    dtype = dereftype(pr, args)
    if dtype <: AbstractSparseArray
        apriori_type = sparsify(apriori_type, true)
    end
    return apriori_type
    # if length(ref_indices(pr)) == 2
    #     if pr isa ParameterRef && dereftype(pr, args) <: AbstractSparseMatrix
    #         return SparseMatrixRef()
    #     else
    #         return MatrixRef()
    #     end
    # else
    #     if pr isa ParameterRef && dereftype(pr, args) <: AbstractSparseVector
    #         return SparseVecRef()
    #     else
    #         return VecRef()
    #     end
    # end
end

# function reftype(pr::Union{Type{ParameterRef}, ParameterRef}, args::Type)
#     if pr isa Type
#         pr = pr()
#     end
#     _dereftype = dereftype(pr, args)

#     if length(ref_indices(pr)) == 2
#         if pr isa ParameterRef && _dereftype <: AbstractSparseMatrix
#             return SparseMatrixRef()
#         else
#             return MatrixRef()
#         end
#     else
#         if pr isa ParameterRef && _dereftype <: AbstractSparseVector
#             return SparseVecRef()
#         else
#             return VecRef()
#         end
#     end
# end

### CONTRACTIONS

"""
Get a reference to a matrix in a contraction
    Returns nothing if it's not found
"""
function matrix_ref(rc::Union{Type{<:RefMult}, RefMult}, args)
    if rc isa Type
        rc = rc()
    end
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
function vec_refs(rc::Union{Type{<:RefMult}, RefMult}, args)
    if rc isa Type
        rc = rc()
    end
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
function struct_ref_exps(ref::RefMult)
    return tuple(Iterators.flatten(struct_ref_exp.(get_prefs(ref)))...)
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

function contraction_type(rc::RefMult, args)
    prefs = get_prefs(rc)
    if length(prefs) == 2
        if reftype(last(prefs), args) == MatrixLike
            return SparseColumn()
        else
            return VectorContraction()
        end
    end
end

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

export @ParameterRefs

include("Resolvers.jl")

# getval(::Type{Val{T}}) where T = T
# getval(::Val{T}) where T = T
