# mutable struct ValueExpression{T}
#     name::Symbol
#     val::Expr

#     indexes::Tuple{Vararg{Union{Symbol, Int}}}
#     flag::Any
# end

# function ValueExpression(val, indexes::Tuple{Vararg{Union{Symbol, Int}}} = (); name = gensym(:val), flag::Any = nothing)
#     ValueExpression{typeof(expr)}(name, assignment, expr, indexes, flag)
# end

# function value_expression(pv::Union{Type{<:ParamTensor}, ParamTensor}, name::Symbol)
#     if !isactive(pv)
#         return ValueExpression(:($(default(pv))), tuple())
#     end
#     return ValueExpression(:, true)
# end



