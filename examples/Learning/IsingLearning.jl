using SparseArrays, FileIO, JLD
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
# export blockReadMatrix
function blockReadMatrix(sim::IsingSim, groupsize = 3, border = 8; imsize = 28)
  g = sim.g
  first_row = vcat(repeat([0], border*g.N+border),
                  repeat([1/8],groupsize), repeat([0], g.N-groupsize),
                  repeat([1/8],groupsize ÷ 2), [0], repeat([1/8],groupsize ÷ 2), repeat([0], g.N-groupsize),
                  repeat([1/8],groupsize))
  first_row = vcat(first_row, repeat([0], g.size-length(first_row)) )

  sparse = SparseVector(first_row)
  shift = 0
  for pix in 2:imsize^2 #Do for every mnist pixel
    if mod(pix,imsize) == 1 #If next pixel row, then shift array by dimension of Ising model, otherwise groupsize
      shift += g.N*groupsize - (imsize-1)*groupsize
    else
      shift += groupsize
    end
    
    sparse = hcat(sparse, circshift(first_row,shift))
  end
  return transpose(sparse)
end

# Reads blocks of groupsize*groupsize, ignoring the inner pixel
# export blockRead
function blockRead(sim, groupsize = 3, border = 8; imsize = 28)
  g = sim.g
  return transpose(reshape( (blockReadMatrix(sim, groupsize, border; imsize) * g.state) , imsize, imsize ))
  # return blockReadMatrix(sim, groupsize, border; imsize) * g.state
end

function readOut(sim, groupsize = 3, border = 8; dt = 0.1, t = 2, imsize = 28)
  g = sim.g
  readMatrix = zeros(imsize,imsize)
  for time in dt:dt:t
    state = copy(g.state)

    readMatrix += 1/length(dt:dt:t) * blockRead(sim, groupsize, border; imsize)
    
    sleep(dt)
  end

  return readMatrix
end

# Returns a welsch pattern from a few set ones by from an index number
function welschPatterns(num)
  p1 = [(x <= imsize/2 ? 1 : -1) for x in 1:imsize, y in 1:imsize]
  p2 = [(y <= imsize/2 ? 1 : -1) for x in 1:imsize, y in 1:imsize]

  patterns = [p1,p2]

  return patterns[num]
end

function readIdxs(sim)
  idxs = []
  readoutMatrix = blockReadMatrix(sim)
  for row in 1:(size(readoutMatrix)[1])
    for (idx, el) in enumerate(row)
      if el == 1
        append!(idxs,idx)
      end
    end
  end
  return idxs
end

function readPixelIdxs(sim,idx, groupsize = 3, border = 8)
  g = sim.g
  startIdx = border*g.N + border
  idxs = [startIdx, startIdx+1, startIdx+2, startIdx+g.N , startIdx+g.N + 2 , startIdx+2*g.N,startIdx+2*g.N+1, startIdx+2*g.N+2]
  shift = groupsize * ( (idx-1) % 28) + g.N * (idx ÷ 28)
  return idxs .+ shift
end

function clampPixelIdxs(sim,idx, groupsize = 3, border = 8)
  g = sim.g
  startIdx = (border+1)*(g.N) + border
  shift = groupsize * ( (idx-1) % 28) + g.N * (idx ÷ 28)
  return startIdx + shift
end

# Calculate the overlap with welsch patterns of a imsize*imsize image
# Returns a vector with the overlaps
function welschOverlap(sim, groupsize = 3, border = 8; imsize = 28)
  overlaps = [sum(welschPatterns(i) .* blockRead(sim, groupsize, border; imsize)) for i in 1:2]
end

# Guess WelshIndex
function guessWelschIdx(sim, groupsize = 3, border = 8; imsize = 28)
  findfirstmax(welschOverlap(sim, groupsize, border; imsize))
end

# Converts welsh index to label
function idxToLab(idx)
  map = [4,8]
  return map[idx]
end

# Returns guess of mnist 
function guessLab(sim, groupsize = 3, border = 8; imsize = 28)
  return idxtolab(guessWelschIdx(sim, groupsize, border, imsize))
end

# Returns mnist images for a given label
function mnistByLabel(num)
  return [ mnist["img"][:,:,i] for i in 1:length(mnist["img"][1,1,:]) if mnist["lab"][i] == num ]
end

# Returns list idxs of mnist for a given label
function mnistByLabelIdx(num)
  return [ i for i in 1:length(mnist["img"][1,1,:]) if mnist["lab"][i] == num ]
end



function equiPropUpdate(sim, lab, beta)
  return
end


# Helper functions

# Finds idx of first maximal number
function idxfirstmax(arr)
  el = arr[1]
  maxidx = 1
  for idx in 2:length(arr)
    new_el = arr[idx]
    if new_el > el
      el = new_el
      maxidx = idx
    end
  end
  return idx
end