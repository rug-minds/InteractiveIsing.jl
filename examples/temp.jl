function test(f::Function, i::Int, x::Int)
    1+f(i,x)
end

test(2,3) do x,y
    x^y
end