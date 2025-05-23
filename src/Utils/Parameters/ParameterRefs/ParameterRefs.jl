export ParameterRef, RefMult, SparseAdj, get_prefs, ref_indices, fixed_symb, ref_symbs, is_paramref, ParameterRef, find_paramref, substitute_paramval, @ParameterRefs, ParamCollection

abstract type AbstractParameterRef end

include("ExpressionTools.jl")
include("ExpressionTree.jl")
include("Utils.jl")
include("RefMap.jl")
include("IndexManipulation.jl")

abstract type FType end
struct IdentityF <: FType end
struct Unary <: FType end
struct Binary <: FType end

"""
Every AbstractParameterRef implements
ref_indices : gives tuple with free indices
get_prefs : gives tuple with all refs
issparse : Says wether the ref points to a sparse structure
@generated function (::AbstractParameterRef)(freeindices) ?
expand_exp ?
"""


type_apply(f, apr::Type{<:AbstractParameterRef}) = f(apr())

### PARAMTER REF
struct ParameterRef{Symb, indices, F, D} <: AbstractParameterRef 
    data::D
end

function ParameterRef(symb)
    sstr = String(symb)
    u_found = findfirst(x -> x == '_', sstr)
    if length(sstr) == 2
        return ParameterRef(Symbol(sstr[1]); func = identity, data = nothing)
    end
    get_symb = sstr[1:u_found-1]
    get_index = sstr[u_found+1:end]
    return ParameterRef(Symbol(get_symb), Symbol.(tuple(get_index...))...)
end

function ParameterRef(symb, indices...; func = identity, data = nothing)
    # F = isnothing(func) ? () : func
    return ParameterRef{Symbol(symb), (indices...,), func, typeof(data)}(data)
end

"""
Type forward for generated
"""
ParameterRef{A,B,C,D}() where {A,B,C,D} = ParameterRef{A,B,C,Nothing}(nothing)

setF(pr::ParameterRef, func) = ParameterRef(ref_symb(pr), ref_indices(pr)...; func = func, data = pr.data)

#### RUNTIME STUFF
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
full_symb(pr::ParameterRef) = Symbol(ref_symb(pr), ref_indices(pr)...)
full_symb(::Type{PR}) where PR<:AbstractParameterRef = full_symb(PR())

ref_indices(::ParameterRef{S, idxs}) where {S, idxs} = idxs #Get the indices
"""
Get a set of the indices present in the type of an AbstractParameterRef
"""
ref_indices(::Type{PR}) where PR<:AbstractParameterRef = ref_indices(PR())

"""
For unified syntax to do contractions
"""
function ref_indices_val(ar::AbstractParameterRef)
    t = Val.(ref_indices(ar()))
    return t
end

function ref_indices(apr::AbstractParameterRef)
    t = tuple(union(ref_indices.(get_prefs(apr()))...)...)
    return t
end

## OVERALL FUNCTIONS
getF(::ParameterRef{S, idxs, F}) where {S, idxs, F} = F
getF(::Type{PR}) where PR<:AbstractParameterRef = getF(PR())

remF(pr::AbstractParameterRef) = setF(pr, identity)

function wrapF(apr::AbstractParameterRef, exp = :(@$))
    F = getF(apr)
    if F == identity
        return exp
    else
        return partialf_exp(F, exp)
    end
end

struct_ref_exp(::Type{PR}) where PR<:AbstractParameterRef = struct_ref_exp(PR())

## FLAT PREFS

function flatprefs(apr::AbstractParameterRef, paramrefs = ParameterRef[])
    prefs = get_prefs(apr)
    for pref in prefs   
        if pref isa ParameterRef
            push!(paramrefs, pref)
        else
            flatprefs(pref, paramrefs)
        end
    end
    return tuple(paramrefs...)
end

#### OVERALL FUNCTIONS
Base.sqrt(apr::AbstractParameterRef) = setF(apr, sqrt)
Base.log(apr::AbstractParameterRef) = setF(apr, log)
Base.exp(apr::AbstractParameterRef) = setF(apr, exp)
Base.:^(apr::AbstractParameterRef, power::Real) = setF(apr, PartialF(^, nothing, power))
Base.:^(power::Real, apr::AbstractParameterRef) = setF(apr, PartialF(^, power, nothing))
Base.:/(apr::AbstractParameterRef, factor::Real) = setF(apr, PartialF(/, nothing, factor))
Base.:/(factor::Real, apr::AbstractParameterRef) = setF(apr, PartialF(/, factor, nothing))
Base.:*(apr::AbstractParameterRef, factor::Real) = setF(apr, PartialF(*, nothing, factor))
Base.:*(factor::Real, apr::AbstractParameterRef) = setF(apr, PartialF(*, factor, nothing))


@generated function get_ref(p, args::DataType)
    return :(args.$(ref_symb(p)))
end

#Expression stuff
function getzero_exp(apr::AbstractParameterRef, precision = nothing)
    if isnothing(precision)
        return :(zero(promote_eltype($(struct_ref_exp(apr)...))))
    else    
        return :(zero($(precision)))
    end
end

function getzero_exp(precision::Union{Nothing, Type}, names...)
    if !isnothing(precision)
        return :(zero($(precision)))
    else
        return :(zero(promote_eltype($(names...))))
    end
end

"""
Get the reference to the struct in either args or params
    Based on the symbol
"""
function struct_ref_exp(p::ParameterRef)
    path = tuple(refmap(Val(ref_symb(p)))...)
    return (build_getproperty_chain(:args, path),)
end

"""
Get the substructs that are referenced
    I.e. if we have args.params.s1.s2.s3
    and the ref is s1_i_j
    then the substructs are (args.params.s1, args.params.s1.s2)
    TODO: Check if this is even true
"""
function substructs(p::ParameterRef)
    if length(struct_ref_exp(p)) == 1
        return nothing
    else
        return struct_ref_exp(p)[1:end-1]
    end
end

"""
Get the indices to be filled in as expression e.g. "[i]"
"""
function  struct_ref_idx_exp(p::ParameterRef)
    return :([$(ref_indices(p)...)])
end

isactive(::Any) = true

function Base.getindex(pr::AbstractParameterRef, idx)
    if pr isa ParameterRef
        error("Cannot index a ParameterRef")
    end
    get_prefs(pr)[idx]
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

expand_exp(pr::Type{<:AbstractParameterRef}) = expand_exp(pr())
function expand_exp(pref::ParameterRef{S, idxs, f}) where {S, idxs, f}
    if isempty(f)
        return Expr(:ref, struct_ref_exp(pref)..., ref_indices(pref)...)
    else
        return Expr(:call, f[1], Expr(:ref, struct_ref_exp(pref)..., ref_indices(pref)...), f[2])
    end
end

@inline function intersect_indices(pr::AbstractParameterRef, idxs)
    idxs[ref_indices(pr)]
end

## ACCESSORS
@generated function Base.get(pr::ParameterRef, args)
    ref = struct_ref_exp(pr)
    exp = quote $(ref...) end
    return exp
end
export get

num_free(apr::AbstractParameterRef) = length(ref_indices(apr))

"""
Go from parameter ref to the struct it wants to reference
"""
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
    # return gettype_recursive(args, struct_refs)
    return gettype_recursive_nongen(args, struct_refs)
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


loopconstant(::Any) = false
### Getting the ref
function generate_block(reftype::ParameterRef, argstype, idxs = (;), precision = nothing, assignments = nothing)
    ind = ref_indices(reftype)
    filled_indices = index_names(idxs)
    contract_ind = idx_subtract(ind, filled_indices)

    struct_ref = struct_ref_exp(reftype)
    rtype = dereftype(reftype, argstype)
    func_exp = partialf_exp(getF(reftype), :(@$))
    expr = dimsum_exp(rtype, filled_indices; valname = struct_ref[1], fexp = func_exp)
    
    # rtype = dereftype(reftype, argstype)
    # func = getF(reftype)
    # vec_index_exp = Expr(:ref, :vec, ind...)
    # # value_exp = expr_F_wrap(reftype, vec_index_exp)
    # value_exp = Expr(:call, func, vec_index_exp)

    # totalname = gensym(:total)
    # expr = nothing
    # if loopconstant(rtype)
    #     expr = :($totalname += $(unroll_exp(rtype, :vec, name -> expr_F_wrap(reftype, name))))
    # else
    #     expr = nested_turbo_wrap(:( $totalname += $(value_exp)), (:(axes(vec, $i_ind)) for i_ind in 1:length(contract_ind)) |> collect, contract_ind)
    # end
    

    # expr = quote
    #     $(unpack_keyword_expr(filled_indices, :idxs))
    #     vec = $(struct_ref_exp(reftype)...)
    #     $totalname = eltype(vec)(0)
    #     $(expr)
    #     # rtype = $(rtype)
    #     $totalname
    # end

    return expr
end

paramref_type_exp = nothing
paramref_type_args = []

(pr::ParameterRef)(args::AS; idxs...) where AS = @inline (pr)(args, (;idxs...))

@inline @generated function (pr::ParameterRef)(args::AS, idxs) where AS
    global paramref_type_exp = generate_block(pr(), args, idxs, rem_lnn = false)
    return paramref_type_exp
end

include("Simplify.jl")

include("RefReduce.jl")
include("RefMult.jl")

include("RefTree.jl")
include("Traits.jl")

include("Blocks.jl")

include("BlockModels.jl")
include("Resolvers.jl")
include("Show.jl")
