function a()
    return b()
end



@generated function c()
    s = invokelatest(a)
    return :( $s * " from generated function!" )
end

function b()
    return "Hello, World!"
end

