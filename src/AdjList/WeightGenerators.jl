struct WeightGeneratorNew{NN}
    func::Function
    funcexp::Union{String, Expression, Nothing}
    rng::Random.AbstractRNG
end

function WeightGeneratorNew(func, NN = tuple(), rng = Random.MersenneTwister())
    if func isa Expression
        f = eval(func)
    elseif func isa String
        f = Meta.parse(func) |> eval
        exp = func
    elseif func isa Function
        f = func
        exp = nothing
    end
    new{NN}(f, exp, rng)
end







