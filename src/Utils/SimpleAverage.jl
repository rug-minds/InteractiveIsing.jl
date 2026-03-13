mutable struct SimpleAverage{T}
    sum::T
    count::Int
end
SimpleAverage(T) = SimpleAverage{T}(0,0)
Base.push!(avg::SimpleAverage, val) = (avg.sum += val; avg.count += 1)
avg(avg::SimpleAverage) = avg.sum/avg.count
reset!(avg::SimpleAverage) = (avg.sum = 0; avg.count = 0)