
function showexpr(rr::RefReduce)
    red_fs = get_fs(rr)
    wrapF(rr, Expr(:call, :+, showexpr(rr[1]), (red_fs[i] == (+) ? showexpr(rr[i+1]) : Expr(:call, :-, showexpr(rr[i+1])) for i in 1:length(red_fs))...))
end

function showexpr(rm::RefMult)|
    mult_fs = get_fs(rm)
    wrapF(rm, Expr(:call, nameof(mult_fs), (showexpr(rm[i]) for i in 1:length(rm))...))
    # Expr(:call, :*, (wrapF(rm[i], showexpr(rm[i])) for i in 1:length(rm))...)
end

function showexpr(pr::ParameterRef)
    symb = ref_symb(pr)
    inds = ref_indices(pr)
    return wrapF(pr, Symbol(symb,"_" ,inds...))
end

function Base.show(io::IO, apr::AbstractParameterRef)
    print(io, "@PR " , showexpr(apr))
end

