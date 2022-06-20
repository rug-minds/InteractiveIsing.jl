module CreateFunc

ftmp = identity

struct Runnable
    x::Vector{Float64}
    y::Vector{Float64}
end

function (r::Runnable)(n)
    r.y .= ftmp.(r.x)
end

function setfn(exstr)
    Core.eval(CreateFunc,Meta.parse("ftmp = $(exstr)"))
end

end