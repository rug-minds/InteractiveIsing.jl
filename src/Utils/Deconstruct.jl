@inline deconstruct(nt::NamedTuple) = nt

@inline @generated function deconstruct(s::S) where {S}
    fnames = fieldnames(S)
    nt_expr = Expr(
        :tuple,
        Expr(
            :parameters,
            (Expr(:kw, name, :(getproperty(s, $(QuoteNode(name))))) for name in fnames)...,
        ),
    )
    return nt_expr
end
export deconstruct
