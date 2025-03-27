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
struct RefMult{Refs, idxs, F, D} <: AbstractParameterRef 
    data::D
end
"""
Stndard Constructor
"""
function RefMult{Refs, idxs}(data = nothing; func = tuple()) where {Refs, idxs}
    RefMult{Refs, idxs, func, typeof(data)}(data)
end
"""
Empty constructor for generated functions
"""
RefMult{A,B,C,D}() where {A,B,C,D} = RefMult{A,B,C,Nothing}(nothing)

"""
Multiplications of two refs
"""
Base.:*(p1::AbstractParameterRef, p2::AbstractParameterRef) = RefMult(p1, p2)

"""
Iterating over a RefMult
"""
Base.iterate(rc::RefMult, state = 1) = iterate(get_prefs(rc), state)

"""
Get overall function
"""
get_F(rm::RefMult{Refs, idxs, fs}) where {Refs, idxs, fs} = fs

"""
Is overall function Unary (stored as (F)) or Binary, stored as (F, number)
"""
F_type(rm::AbstractParameterRef) = length(get_F(rm)) == 1 ? Unary() : Binary()

"""
Overall function for RefMult
"""
Base.:-(rm::RefMult{Refs, idxs}) where {Refs,idxs} = RefMult{Refs, idxs}(rm.data, func = tuple(-))

"""
Overall function for RefMult
"""
Base.:^(rm::RefMult{Refs,idxs}, pow::Real) where {Refs,idxs} = RefMult{get_prefs(rm), getidxs(rm)}(rm.data, func = (^, pow))

"""
Get the return type of the RefMult
"""
function return_type(rm::RefMult{Refs, idxs, F, D}, args) where {Refs, idxs, F, D}
    promote_type(return_type.(Refs, Ref(args))...)
end


"""
Get all the symbols that are in the refmult (symbols refer to the name of the parameter)
"""
ref_symb(::RefMult{Refs}) where Refs = tuple(ref_symb.(Refs)...)


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
issparse(::RefMult{Refs, idxs}) where {Refs, idxs} = all(issparse.(Refs))

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
    @assert !mixed_mult(rm)
    return tuple(Iterators.flatten(struct_ref_exp.(Refs))...)
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


"""
Check if a RefMult is pure
    This only holds if it is a simple multiplication of the refs
    And all the underlying refs are pure
    I.e. (a_i + b_j) * (c_i + d_j) is not pure, 
        even though both reduces have the same indices
"""
@generated function ispure(rm::RefMult{Refs}) where {Refs}
    if typeof(Refs) <: Tuple{Vararg{Tuple}} # If Refs has multiple partitions, then it is not pure
        return :(false)
    end
    pure = true
    # indexset = ref_indices(rm())
    prefs = get_prefs(rm())
    first_inddexset = ref_indices(first(prefs))
    for ref in prefs
        idxs = ref_indices(ref)
        pure = ispure(ref) && idxs ∈ first_inddexset
        if !pure
            break
        end
    end
    return :($pure)
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
    if !(filled_indices <: Nothing) # If indices are filled, they are esentially not there
        _intersect = tuple((setdiff(_intersect, getval(filled_indices)))...)
    end
    return :($_intersect)
end

"""
Group two parameter refs into a RefMult
"""
function group_mults(p1::AbstractParameterRef, p2::AbstractParameterRef)
    if ispure(p1) && ispure(p2) && free_symb(p1) == free_symb(p2)
        return RefMult{tuple(p1,p2), free_symb(p1)}(p1.data)
    else
        return RefMult{tuple(tuple(p1), tuple(p2)), tuple(indices_set(p1,p2)...)}(p1.data)
    end
end

"""
Create a RefMult from two parameter refs
"""
function RefMult(p1::AbstractParameterRef, p2::AbstractParameterRef)
    return group_mults(p1, p2)
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
function expand_exp(rc::RefMult{Refs, idxs}) where {Refs, idxs}
    @assert !mixed_mult(rc)
    exp = Expr(:call, :*, expand_exp.(get_prefs(rc))...)
    return expr_F_wrap(rc, exp)
end
