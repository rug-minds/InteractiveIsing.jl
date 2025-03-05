export ParameterRef, RefContraction, SparseAdj, get_prefs, free_symb, fixed_symb, ref_symbs, is_paramref, ParameterRef, find_paramref, substitute_paramval, @ParameterRefs, ParamCollection

abstract type AbstractParameterRef end
struct ParameterRef{Symb, symbs} <: AbstractParameterRef end
struct RefContraction{L, R, index} <: AbstractParameterRef end

Base.iterate(rc::RefContraction, state = 1) = iterate(get_prefs(rc), state)

# abstract type ContractionType end
# struct SparseAdj <: ContractionType end

abstract type PRefType end
struct PSparseAdj <: PRefType end
struct PVec <: PRefType end

abstract type ContractionType end
struct SparseRowContraction <: ContractionType end
struct SparseContraction <: ContractionType end
struct VectorContraction <: ContractionType end


## ACCESSORS

"""
For unified syntax to do contractions
"""
get_prefs(p::ParameterRef) = tuple(p)
get_prefs(p::RefContraction{Refs}) where Refs = Refs

get_ref_symb(::ParameterRef{S}) where S = S
ref_symbs(::RefContraction{Refs}) where Refs = tuple(get_ref_symb.(Refs)...)

free_symb(::ParameterRef{S, fs}) where {S, fs} = fs
free_symb(::RefContraction{Refs, contractions}) where {Refs, contractions} = tuple(symb for symb in (free_symb.(Refs)...) if !(symb ∈ contractions))
free_symb(::Type{PR}) where PR<:AbstractParameterRef = free_symb(PR())

fixed_symb(::ParameterRef{S, free_symb, fixed_symb}) where {S, F, free_symb, fixed_symb} = fixed_symb
fixed_symb(::RefContraction{Refs, contractions}) where {Refs, contractions} = union(fixed_symb.(Refs)...)
fixed_symb(::Type{PR}) where PR<:AbstractParameterRef = fixed_symb(PR())


## PREF FUNCTIONS
"""
Get the ref type for a parameter ref
"""
function reftype(pr::ParameterRef)
    if length(free_symb) + length(fixed_symb) == 2
        return PSparseAdj
    else
        return PVec
    end
end


"""
Get the reference to the struct in either args or params
    Based on the symbol
"""
function struct_ref_exp(p::ParameterRef)
    if get_ref_symb(p) == :w
        return :(args.gadj)
    elseif get_ref_symb(p) == :s
        return :(args.gstate)
    else
        return :(params.$(get_ref_symb(p)))
    end
end

struct_ref_exp(::Type{PR}) where PR<:AbstractParameterRef = struct_ref_exp(PR())


"""
Get the indices to be filled in as expression e.g. "[i]"
"""
function  struct_ref_idx_exp(p::ParameterRef)
    return :([$(join(fixed_symb, ","))])
end


### CONTRACTIONS

"""
Get a reference to a matrix in a contraction
"""
function adj_ref(rc::RefContraction)
    _refsymb = nothing
    for ref in rc
        if reftype(ref) == PSparseAdj
            _refsymb = get_ref_symb(ref)
            break
        end
    end
    return _refsymb
end

function vec_refs(rc::RefContraction)
    refs = get_prefs(rc)
    return (_vec_refs(gethead(refs), gettail(refs))...)
end

function _vec_refs(head, tail)
    if reftype(head) == PVec
        return head, _vec_refs(gethead(tail), gettail(tail))
    else
        return _vec_refs(gethead(tail), gettail(tail))
    end
end

"""
Get all references to the structs in a contraction
"""
function struct_ref_exps(ref::RefContraction)
    return struct_ref_exp.(get_prefs(ref))
end


function RefContraction(p1::AbstractParameterRef, p2::AbstractParameterRef)
    f1 = free_symb(p1)
    f2 = free_symb(p2)
    refs = union(get_prefs(p1), get_prefs(p2)) 
    overlap = intersect(f1, f2)
    return RefContraction{tuple(refs...), typeof(overlap)}()
end

function getref(args, params, ::Val{R}) where R
    if R == :w
        return args.gadj
    elseif R == :s
        return args.gstate
    else
        return params.R
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

function reducerefs(refs, type, contraction_idx)
    return type(_reducerefs(Processes.gethead(refs), Processes.gettail(refs), type, contraction_idx))
end

function _reducerefs(head, tail, type, contraction_idx)
    return type(head[contraction_idx]) + _reducerefs(Processes.gethead(tail), Processes.gettail(tail), type, contraction_idx)
end

_reducerefs(::Nothing, ::Any, ::Any, ::Any) = 0


Base.:*(p1::ParameterRef, p2::ParameterRef) = RefContraction(p1, p2)


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
    fixed_index = tuple(Symbol.(tuple(get_index...))[1])
    free_index = Symbol.(tuple(get_index...))[2:end]
    return ParameterRef{Symbol(get_symb), free_index, fixed_index}()
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

macro ParameterRefs(ex)
    @capture(ex, function fname_(a__) body_ end)
    println("Body: ", body)
    symbs = find_paramref(body)
    println("Symbs: ", symbs)
    for (symb, indexes) in zip(symbs[1], symbs[2])
        body = substitute_paramref(body, indexes, symb)
    end
    println(ex)
end


d_ex = quote function Δi_H(::Ising)
        contractions = :(w_ij*s_j)
        multiplications = :((s_i^2-sn_i^2)*self_i+(s_i-sn_i)*(b_i))
        return (;contractions, multiplications)
    end
end

export d_ex, @ParameterRefs
ev =  @ParameterRefs function ΔiH(::Ising)
    contractions = :(w_ij*s_j)
    multiplications = :((s_i^2-sn_i^2)*self_i+(s_i-sn_i)*(b_i))
    return (;contractions, multiplications)
end

##

reduce_contraction_exp = nothing

@generated function reduce_contraction(rc::RefContraction, args, params, ::SparseRowContraction)
    global reduce_contraction_exp = quote
        i = args.i
        cumsum = zero(promote_eltype($(join(struct_refs(rc), ", "))))
        sp_adj = adj_ref(rc)
        @turbo for ptr in nzrange(sp_adj, i)
            j = sp_adj.rowval[ptr]
            wij = sp_adj.nzval[ptr]
            cumsum += wij * 
        end
        return cumsum
    end
    return reduce_contraction_exp
end