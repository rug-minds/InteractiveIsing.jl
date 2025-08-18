@inline function func_tuple_unroll(fs, args::Union{Nothing, Tuple})
    tuple(_func_tuple_unroll(gethead(fs), gettail(fs), args)...)
end

@inline function _func_tuple_unroll(fhead, ftail, args)
    if isnothing(args)
        if isempty(ftail)
           return (fhead(),)
        else
           return (fhead(), _func_tuple_unroll(gethead(ftail), gettail(ftail), args)...)
        end
    else
        if isempty(ftail)
            return (fhead(args...),)
        else
            return (fhead(args...), _func_tuple_unroll(gethead(ftail), gettail(ftail), args)...)
        end
    end
end

