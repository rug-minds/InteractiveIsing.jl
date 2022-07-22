__precompile__()

module IsingLearning
push!(LOAD_PATH, pwd())

using IsingGraphs, MLDatasets

export read_idx, clampIm, mnist_im, mnist_lab

# const idx_types = [(UInt8, 0x08), (Int8, 0x09), (Int16, 0x0B), (Int32, 0x0C), (Float32, 0x0D), (Float64, 0x0E)]
# const typefromcode = Dict(idx_types)
# const codefromtype = Dict(map(reverse, idx_types))

# function write_idx(stream::Union{IOStream,String}, data::Array{T,N}) where {T,N}
#   dims = convert(Array{Int32}, collect(size(data)))
#   write(stream, 0x00, 0x00, typefromcode[T], UInt8(N), hton.(dims), hton.(data))
# end

# read_idx(filename::String) = open(read_idx, filename)

# function read_idx(stream::IOStream)
#   header = read(stream, 4)
#   dims = ntoh.(read!(stream, Array{Int32}(undef, header[4])))
#   data = ntoh.(read!(stream, Array{codefromtype[header[3]]}(undef, dims...)))
# end

# mnist_im = read_idx(open("Learning/Dataset/train-images.idx3-ubyte"))
# mnist_ex = read_idx(open("Learning/Dataset/train-labels.idx1-ubyte"))

# function mnist_img(num)
#   mToI.(mnist[num, :,:])
# end

mnist = MNIST()

function mnist_im(num, crange = (0,1))
  (crange[2]-crange[1]) .* mnist[num][1] .+ crange[1]
end

function mnist_lab(num)
  mnist[num][2]
end

# Mnist value to ising value
@inline function mToI(int::UInt8, irange = (0,1))::Float32
  return irange[1]+(irange[2]-irange[1])/256*int
end

# # Ising value to mnist value
# @inline function iToM(num::Float32, irange = (0,1))::UInt8
#   return UInt8(floor((num+1)*256/2)+1)
# end



# Groupsize should be odd?
# Crange: Clamp range
function clampIm(g::IsingGraph{Float32}, num::Integer,groupsize=5; crange = (0,1))
  while groupsize*28 > g.N || iseven(groupsize) && groupsize > 0
    groupsize -= 1
  end

  border = Int32(round((g.N - groupsize*28)/2))
  println("Bordersize $border")

  clamp_region_length = Int32((g.N-border*2)/groupsize)

  # println(groupsize)

  idxs = [Int16.((border+i*groupsize,border+j*groupsize)) for i in 1:(clamp_region_length), j in 1:(clamp_region_length)]

  idxs_array::Array{Int32} = reshape(transpose(coordToIdx.(idxs,g.N)), length(idxs))
 
  img_is = mnist_im(num, crange)

  brushs = img_is[:]
  println("Size of brushs $(length(brushs)), size of idxs $(length(idxs_array))")

  setSpins!(g, idxs_array, brushs, true)

end

function clampMag(g::IsingGraph{Float32}, num::Integer, groupsize = 5, crange = (0,1))
  while groupsize*28 > g.N || iseven(groupsize) && groupsize > 0
    groupsize -= 1
  end

  border = Int32(round((g.N - groupsize*28)/2))
  println("Bordersize $border")

  clamp_region_length = Int32((g.N-border*2)/groupsize)
end