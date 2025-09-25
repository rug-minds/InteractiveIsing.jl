using InteractiveIsing
using FileIO, DelimitedFiles, GLMakie

gr = IsingGraph(precision = Float64, architecture= [(15,16, Continuous),(24,24,Discrete)], sets = [(0,1),(0,1)])


function readnumbers(filename)
    return reshape(readdlm(filename, Int64), (2000, 15,16))./6
end

function getnumber(data, num)
    return data[num,:,:]
end

function plotnumber(data, num)
    Fa = plot(data[num,:,:])
    ax = Fa.axis
    ax.yreversed = true
    return Fa
end

function normalize_numbers(numbers, layer::IsingLayer{Continuous, SS}) where SS
    #Normalize
    ss_dist = last(SS)-first(SS)
    numbers .*= ss_dist
    numbers .+= first(SS)
    return numbers

end

numbers = readnumbers("examples/Learning/numbers.txt")
normalize_numbers(numbers, gr[1])


simulate(gr)

function clampnum(layer::IsingLayer{Continuous,SS}, numbers, idx::Int) where SS
    setSpins!(layer, getnumber(numbers,idx), true)
end

genAdjFull!(gr[1], gr[2])

clampnum(gr[1], numbers, 6)

