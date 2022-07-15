__precompile__()

module IsingLearning

using ..IsingGraphs

export read_idx, clampIm, mnist_im, mnist_ex

const idx_types = [(UInt8, 0x08), (Int8, 0x09), (Int16, 0x0B), (Int32, 0x0C), (Float32, 0x0D), (Float64, 0x0E)]
const typefromcode = Dict(idx_types)
const codefromtype = Dict(map(reverse, idx_types))

function write_idx(stream::Union{IOStream,String}, data::Array{T,N}) where {T,N}
  dims = convert(Array{Int32}, collect(size(data)))
  write(stream, 0x00, 0x00, typefromcode[T], UInt8(N), hton.(dims), hton.(data))
end

read_idx(filename::String) = open(read_idx, filename)

function read_idx(stream::IOStream)
  header = read(stream, 4)
  dims = ntoh.(read!(stream, Array{Int32}(undef, header[4])))
  data = ntoh.(read!(stream, Array{codefromtype[header[3]]}(undef, dims...)))
end

mnist_im = read_idx(open("Learning/Dataset/train-images.idx3-ubyte"))
mnist_ex = read_idx(open("Learning/Dataset/train-labels.idx1-ubyte"))

# Mnist value to ising value
@inline function mToI(int::UInt8, irange = (0,1))::Float32
  return irange[1]+(irange[2]-irange[1])/256*int
end

# # Ising value to mnist value
# @inline function iToM(num::Float32, irange = (0,1))::UInt8
#   return UInt8(floor((num+1)*256/2)+1)
# end



# Groupsize should be odd?
function clampIm(g::CIsingGraph,mnist::Matrix,num::Integer,groupsize=5)

  rem = rem(g.N,groupsize)
  clamp_region_length = Int32((g.N-rem)/groupsize)
  init_idx = -floor(-groupsize/2)
  idxs = [(init_idx+i*groupsize,init_idx+j*groupsize) for i in 1:(clamp_region_length-1), i in 1:1:(clamp_region_length-1)]

  img_is = transpose(mToI.(mnist[num, :,:]))

  brushs = img_is[:]

  setSpins!(g, idxs, brushs, clamp=true)

end

end