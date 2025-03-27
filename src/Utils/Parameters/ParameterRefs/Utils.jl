"""
Get the expression that unpacks the keyword arguments with var name `name`
    (;k1, k2, ...) = name
"""
function unpack_keyword_expr(kwtuple::NamedTuple, name::Symbol)
    return :((;$(keys(kwtuple)...)) = $name)
end
"""
For a keyword tuple type get the expression that unpacks the keyword arguments
    (;k1, k2, ...) = name
"""
function unpack_keyword_expr(kwtuple::Type{<:NamedTuple}, name::Symbol)
    keys = fieldnames(kwtuple)
    return :((;$(keys...)) = $name)
end

# """
# Old system?
# """
# @generated function symb_intersect(rc::RefMult{Refs}) where Refs
#     rfs = Refs
#     s = []
#     for ref in rfs
#         push!(s, ref_indices(ref))
#     end
#     return :($(tuple(intersect(s...)...)))
# end

# """
# Old System?
# """
# @generated function symb_intersect_val(rc::RefMult{Refs}) where Refs
#     rfs = Refs
#     s = []
#     for ref in rfs
#         push!(s, ref_indices(ref))
#     end
#     return :($(tuple(Val.(intersect(s...))...)))
# end