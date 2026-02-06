function without_val(n, x)
    if n == 3
        return x + 10
    elseif n == 4
        return x * 20
    else
        return x
    end
end

function with_val(::Val{N}, x) where N
    if N == 3
        return x + 10
    elseif N == 4
        return x * 20
    else
        return x
    end
end

wrapped_withval(n, x) = with_val(Val(n), x)

function checklowered_without()
    @code_lowered without_val(3, 5)
end

function checklowered_with()
    @code_lowered with_val(Val(3), 5)
end

function checklowered_wrapped()
    @code_lowered wrapped_withval(3, 5)
end

checklowered_without()
checklowered_with()
checklowered_wrapped()

function benchmark_without()
    cum = zero(Float64)
    for _ in 1:100000
        cum += without_val(3, 5)*rand()
    end
    return cum
end

function benchmark_with()
    cum = zero(Float64)
    for _ in 1:100000
        cum += with_val(Val(3), 5)*rand()
    end
    return cum
end

function benchmark_wrapped()
    cum = zero(Float64)
    for _ in 1:100000
        cum += @inline wrapped_withval(3, 5)*rand()
    end
    return cum
end

@benchmark benchmark_without()
@benchmark benchmark_with()
@benchmark benchmark_wrapped()