function test(a,b)
    return a+b
end

function test(a,b,c)
    return a+b+c
end

function standardargs(f::typeof(test1))

end

function dispatch(tuple)
    test(tuple...)
end

function dispatch(a,b,c)
    test(a,b,c)
end