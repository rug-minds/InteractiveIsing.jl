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

numbers = readnumbers("examples/Learning/numbers.txt")

plotnumber(numbers, 4)

clampImg!(gr,)