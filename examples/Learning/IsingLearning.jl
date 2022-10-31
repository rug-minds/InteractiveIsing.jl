using SparseArrays
# export read_idx, mnist_img, mnist_lab, mnist, clampImg!, clampMag!

const mnist_path = joinpath(@__DIR__, "mnist.jld")
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

  # Always mnist dimension..
  clamp_region_length = 28
  # clamp_region_length = Int32((g.N-border*2)/groupsize)
  
  # println(groupsize)

  idxs = [Int16.((border+i*groupsize,border+j*groupsize)) for i in 1:(clamp_region_length), j in 1:(clamp_region_length)]

  idxs_array::Array{Int32} = reshape(transpose(coordToIdx.(idxs,g.N)), length(idxs))
 
  img_is = mnist_img(num, crange)

  brushs = img_is[:]
  println("Size of brushs $(length(brushs)), size of idxs $(length(idxs_array))")

  setSpins!(g, idxs_array, brushs, true)

  return groupsize, border
end

function clampMag!(sim::IsingSim, num::Integer; groupsize = 5, crange = (0,1))
  g = sim.g
  while groupsize*28 > g.N || iseven(groupsize) && groupsize > 0
    groupsize -= 1
  end

  border = Int32(round((g.N - groupsize*28)/2))
  println("Bordersize $border")

  # Always mnist dimension..
  clamp_region_length = 28
  # clamp_region_length = Int32((g.N-border*2)/groupsize)

  idxs = [Int16.((border+i*groupsize,border+j*groupsize)) for i in 1:(clamp_region_length), j in 1:(clamp_region_length)]

  idxs_array::Array{Int32} = reshape(transpose(coordToIdx.(idxs,g.N)), length(idxs))
 
  img_is = mnist_img(num, crange)

  brushs = img_is[:]
  println("Size of brushs $(length(brushs)), size of idxs $(length(idxs_array))")
  
  remM!(sim)
  setMIdxs!(sim, idxs_array, brushs)

  return groupsize, border
end

# Read 3 by 3 blocks of around clamped pixel and ignores middle pixel. 
# Converts it into image of 28 by 28 (1 to 1 with amount of mnist pixels)
# Every pixel is obtained by averaging all states next to the clamped pixel
# export blockRead
function blockRead(sim::IsingSim, groupsize, border; imsize = 28)
  g = sim.g
  first_row = vcat(repeat([0], border),
                  repeat([1/8],groupsize), repeat([0], g.N),
                  repeat([1/8],groupsize ÷ 2), [0], repeat([1/8],groupsize ÷ 2), repeat([0], g.N),
                  repeat([1/8],groupsize))
  first_row = vcat(first_row, repeat([0], g.size-length(first_row)) )

  sparse = SparseVector(first_row)
  shift = 0
  for pix in 2:imsize^2 #Do for every mnist pixel
    if mod(pix,imsize) == 1 #If next pixel row, then shift array by dimension of Ising model, otherwise groupsize
      shift += g.N
    else
      shift += groupsize
    end
    
    sparse = hcat(sparse, circshift(first_row,shift))
  end
  return sparse
end