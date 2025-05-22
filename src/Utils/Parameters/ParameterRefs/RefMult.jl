"""
Used to store multiplications between parameters
Refs store an set of AbstractParameterRefs
    If refs have different indices, or the underlying refs are not pure they are stored as a tuple of tuples ((R1), (R2))
        We call this a mixed RefMult
    If refs have the same indices they are stored as a single tuple (R1,R2)
        We call this a pure RefMult
        The significance of being pure is that this may be represented simply as
        a simple multiplication of the ref values for a fill and contraction
"""
struct RefMult{Refs, mult_f, F, D} <: AbstractParameterRef 
    data::D
end

get_mult_f(rm::RefMult{Refs, mult_f}) where {Refs, mult_f} = mult_f
get_fs(rm::RefMult) = get_mult_f(rm)

Base.length(rm::RefMult{Refs}) where Refs = length(Refs)
"""
Standard Constructor
"""
function RefMult{Refs, mult_f}(data = nothing; func = identity) where {Refs, mult_f}
    RefMult{Refs, mult_f, func, typeof(data)}(data)
end

function RefMult(refs, mult_f, data = nothing; func = identity)
    RefMult{refs, mult_f, func, typeof(data)}(data)
end

"""
Create a RefMult from two parameter refs
"""
function RefMult(p1::AbstractParameterRef, p2::AbstractParameterRef, f::Function = *)
    if ispure(p1) && ispure(p2) && ref_indices(p1) == ref_indices(p2)
        if f == *
            data = p1.data
            if p1 isa RefMult && getF(p1) == identity && get_fs(p1) == *
                p1 = get_prefs(p1)
            else
                p1 = tuple(p1)
            end
            if p2 isa RefMult && getF(p2) == identity && get_fs(p2) == *
                p2 = get_prefs(p2)
            else
                p2 = tuple(p2)
            end
            return RefMult{tuple(p1...,p2...), f}(data)
        end
        return RefMult{tuple(p1,p2), f}(p1.data)
    else
        return RefMult{tuple(tuple(p1), tuple(p2)), f}(p1.data)
    end
end

"""
Empty constructor for generated functions
"""
RefMult{A,B,C,D}() where {A,B,C,D} = RefMult{A,B,C,Nothing}(nothing)

"""
Multiplications of two refs
"""
Base.:*(p1::AbstractParameterRef, p2::AbstractParameterRef) = RefMult(p1, p2, *)
function Base.:/(p1::AbstractParameterRef, p2::AbstractParameterRef)
    RefMult(p1, p2, /)
end

"""
Iterating over a RefMult
"""
Base.iterate(rc::RefMult, state = 1) = iterate(get_prefs(rc), state)

"""
Get overall function
"""
getF(rm::RefMult{Refs, idxs, fs}) where {Refs, idxs, fs} = fs

"""
Set overall function
"""
function setF(rm::RefMult{Refs, mult_f}, func) where {Refs, mult_f}
    return RefMult(Refs, mult_f, rm.data, func = func)
end

"""
Overall function for RefMult
"""
Base.:-(rm::RefMult{Refs, mult_f}) where {Refs, mult_f} = RefMult{Refs, mult_f}(rm.data, func = tuple(-))

# """
# Overall function for RefMult
# """
# Base.:^(rm::RefMult{Refs,idxs}, pow::Real) where {Refs,idxs} = RefMult{get_prefs(rm), getidxs(rm)}(rm.data, func = (^, pow))

"""
Get the return type of the RefMult
"""
function return_type(rm::RefMult{Refs, mult_f, F, D}, args) where {Refs, mult_f, F, D}
    promote_type(return_type.(Refs, Ref(args))...)
end


"""
Get all the symbols that are in the refmult (symbols refer to the name of the parameter)
"""
ref_symb(::RefMult{Refs}) where Refs = tuple(ref_symb.(Refs)...)

### EXPS

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
Get all references to the structs in a contraction
"""
function struct_ref_exps(ref::RefMult)
    return tuple(Iterators.flatten(struct_ref_exp.(get_prefs(ref)))...)
end

### TRAITS
"""
Refs are represented as a tuple of tuples of AbstractParameterRefs if they are mixed
"""
mixed_mult(rm::RefMult{Refs}) where Refs = typeof(Refs) <: Tuple{Vararg{Tuple}}

"""
Opposite of mixedmult
"""
function simplemult(pm::RefMult{Refs}) where Refs
    if typeof(Refs) <: Tuple{Vararg{Tuple}}
        return false
    end
    return true
end


# Does this one matter?
"""
Gives wether all the refs are sparse
"""
issparse(::RefMult{Refs}) where {Refs} = all(issparse.(Refs))

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
Get the flattened parameter refs
Ie. if Refs = ((R1), (R2)) then get_prefs = (R1, R2)
"""
@generated function get_prefs(p::RefMult{Refs}) where Refs
    if mixed_mult(p())
        c = tuple(collect(Iterators.flatten(Refs))...)
        return :($c)
    else
        return :($Refs)
    end
end

"""
Get the struct reference expressions (i.e. args.params.symobl) for each parameter ref
    in a tuple (args.params.s1, args.params.s2, ...)
"""
function struct_ref_exp(rm::RefMult{Refs}) where Refs
    # @assert !mixed_mult(rm)
    refs = get_prefs(rm)
    return tuple(Iterators.flatten(struct_ref_exp.(refs))...)
end




"""
Check if a RefMult is pure
    This only holds if it is a simple multiplication of the refs
    And all the underlying refs are pure
    I.e. (a_i + b_j) * (c_i + d_j) is not pure, 
        even though both reduces have the same indices
"""
function ispure(rm::RefMult{Refs}) where {Refs}
    # if typeof(Refs) <: Tuple{Vararg{Tuple}} # If Refs has multiple partitions, then it is not pure
    #     return :(false)
    # end
    pure = true
    # indexset = ref_indices(rm())
    prefs = get_prefs(rm)
    for ridx in 1:length(prefs)-1
        if !isempty(ref_indices(prefs[ridx])) || !isempty(ref_indices(prefs[ridx+1]))
            if ref_indices(prefs[ridx]) != ref_indices(prefs[ridx+1])
                pure = false
                break
            end
        end
    end
    # return :($pure)
    return pure
end

"""
Get the indices present in the RefMult
"""
@generated function ref_indices(rc::RefMult{Refs}) where Refs
    t = nothing
    if typeof(Refs) <: Tuple{Vararg{Tuple}}
        t = tuple(union(ref_indices.(get_prefs(rc()))...)...)
    else
        t = tuple(union(ref_indices.(Refs)...)...)
    end
    return :($t)
end

### MULTS
"""
Gives the indices that are present in both refs
"""
@generated function indices_set(ref1::AbstractParameterRef, ref2::AbstractParameterRef, filled_indices = nothing)
    idcs1 = ref_indices(ref1())
    idcs2 = ref_indices(ref2())
    _union = tuple(union(idcs1, idcs2)...)
    if !(filled_indices <: Nothing) # If indices are filled, they are esentially not there
        _union = tuple((setdiff(_union, getval(filled_indices)))...)
    end
    # Return each symbol without duplicates
    return :($_union)
end

"""
Given two refs and indices that are filled, return the indices that are contracted
    I.e. the indices that are present in both refs while subtracting the filled indices
    Filled indices are indices specified by the user to have a singular value
    This allows for contractions of say: a_i*w_i2, where in w_ij j is of constant value 2
"""
@generated function contract_indices(ref1, ref2, filled_indices = nothing)
    idcs1 = ref_indices(ref1())
    idcs2 = ref_indices(ref2())
    _intersect = tuple(intersect(idcs1, idcs2)...)
    if !(filled_indices <: Nothing) && !(filled_indices == @NamedTuple{}) # If indices are filled, they are esentially not there
        _intersect = tuple((setdiff(_intersect, getval(filled_indices)))...)
    end
    return :($_intersect)
end

"""

"""
function loop_idxs(ref, filled_indices = nothing)
    indxs = ref_indices(ref)
    filled_indices = index_names(filled_indices)
    if !(isnothing(filled_indices))
        indxs = tuple((setdiff(indxs, filled_indices))...)
    end
    return indxs
end

"""
Expand the expression of the leftmost ref
"""
expand_left(rc::RefMult) = expand_exp(first(get_prefs(rc)))

"""
Expand the total expression of the refmult
    This gives expand calles downwards
    For the refmult this essentially means we get :(F(expand(R1))*F(Expand(R2))*...)
"""
function expand_exp(rc::RefMult{Refs}) where {Refs}
    # @assert !mixed_mult(rc)
    exp = Expr(:call, :*, expand_exp.(get_prefs(rc))...)
    return expr_F_wrap(rc, exp)
end

