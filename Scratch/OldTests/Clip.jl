using BenchmarkTools
function clip1(x,lower, upper)
    x = upper-x;
    x = (x+abs(x))*0.5;
    x = upper-x-lower;
    x = (x+abs(x))*0.5;
    x += lower;
    return x
end

function clip2(x, lower, upper)
    if x < lower
        return lower
    elseif x > upper
        return upper
    else
        return x
    end
end

x = 0.3
@benchmark y = clip1(x, -1.0, 1.0)
@benchmark y = clip2(x, -1.0, 1.0)