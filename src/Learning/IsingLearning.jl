export read_idx, mnist_img, mnist_lab, mnist, clampImg!, clampMag!

const mnist_path = joinpath(@__DIR__, "src", "Learning", "mnist.jld")
const mnist = load(mnist_path)

function mnist_img(num, crange = (0,1))
  (crange[2]-crange[1]) .* mnist["img"][:,:,num] .+ crange[1]
end

function mnist_lab(num)
  mnist["lab"][num]
end

# Mnist value to ising value
@inline function mToI(int::UInt8, irange = (0,1))::Float32
  return irange[1]+(irange[2]-irange[1])/256*int
end

# Groupsize should be odd?
# Crange: Clamp range
function clampImg!(g::IsingGraph{Float32}, num::Integer,groupsize=5; crange = (0,1))
  while groupsize*28 > g.N || iseven(groupsize) && groupsize > 0
    groupsize -= 1
  end

  border = Int32(round((g.N - groupsize*28)/2))
  println("Bordersize $border")

  clamp_region_length = Int32((g.N-border*2)/groupsize)

  # println(groupsize)

  idxs = [Int16.((border+i*groupsize,border+j*groupsize)) for i in 1:(clamp_region_length), j in 1:(clamp_region_length)]

  idxs_array::Array{Int32} = reshape(transpose(coordToIdx.(idxs,g.N)), length(idxs))
 
  img_is = mnist_img(num, crange)

  brushs = img_is[:]
  println("Size of brushs $(length(brushs)), size of idxs $(length(idxs_array))")

  setSpins!(g, idxs_array, brushs, true)

end

function clampMag!(g::IsingGraph{Float32}, num::Integer, groupsize = 5, crange = (0,1))
  while groupsize*28 > g.N || iseven(groupsize) && groupsize > 0
    groupsize -= 1
  end

  border = Int32(round((g.N - groupsize*28)/2))
  println("Bordersize $border")

  clamp_region_length = Int32((g.N-border*2)/groupsize)
end
