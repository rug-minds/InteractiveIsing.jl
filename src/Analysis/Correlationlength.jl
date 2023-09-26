#TODO: Should be called correlation function not sampling

""" 
Takes a random number of pairs for every length to calculate correlation length function
This only works without defects.
"""
# Should all return x, y, e.g. length_bins and correlation
abstract type SamplingAlgorithm end

struct Mtl <: SamplingAlgorithm end
struct CPU <: SamplingAlgorithm end
struct CPU_Sampling <: SamplingAlgorithm end
struct Fourier <: SamplingAlgorithm end

#= 
Correlation length functions should collect the correlation of units from  i - i+1 in a bin, starting from 0.
=#

@static if Sys.isapple()
   using Metal
   include("CorrelationMetal.jl")
   s_algo = Mtl

else
   #Fallback
   s_algo = CPU
end


# Seems not interely accurate for higher distances, why?
"""
Sample correlation length for a given layer with the CPU
Will sample ranodm angles for every length
Doesn't seem to produce entirely accurate results

Took this approach because it can be parallelized easily since every thread
has it's own distances to sample
"""
function correlationLength(layer, ::Type{CPU_Sampling})
   @inline function bin(dist)
      floor(Int32, dist)
   end 

   @inline function randvec(len)
      theta = rand() * 2 * pi
      round.(Int32,[len*cos(theta), len*sin(theta)])
   end
   # percentage of pairs to sample per thread
   pairs = 100000

   state_copy = copy(state(layer))

   l, w = Int32.(size(state_copy))

   mdist = floor(Int32,maxdist(layer))

   bins = zeros(Float32, mdist)

   allidxs = collect(UnitRange{Int32}(1:nStates(layer)))

   avg2 = (sum(state_copy) / (l*w))^2 

   Threads.@threads for len in 1:mdist
      for _ in 1:pairs
         idx1 = rand(allidxs)
         idx2 = sampleDistantState(layer, idx1, len)

         bins[len] += state_copy[idx1] * state_copy[idx2]
      end
   end

   bins = (bins ./ pairs) .- avg2
   return [1:length(bins);], bins
end
export corrCPU

function sampleDistantState(layer, idx1, sdist)
   @inline function randvec(len, theta = nothing)
      if isnothing(theta)
         theta = rand() * 2 * pi
      end
      return (round.(Int32,(len*cos(theta), len*sin(theta))), theta)
   end

   i, j = idxToCoord(idx1, glength(layer))
   (di, dj), theta = randvec(sdist)

   idx2 = coordToIdx(i+di, j+dj, layer)   
end
export sampleDistantState

function correlationLength(layer, ::Type{CPU})
   @inline function bin(dist)
      floor(Int32, dist)
   end 

   _state = copy(state(layer))
   avg_sq = (sum(state(layer))/nStates(layer))^2

   bins = zeros(Float32, floor(Int, maxdist(layer)))
   counts = zeros(Float32, floor(Int, maxdist(layer)))

   for s1 in eachindex(_state)
      for s2 in (s1+1):length(_state)
         _dist = dist(s1, s2, layer)
         bins[bin(_dist)] += _state[s1] * _state[s2]
         counts[bin(_dist)] += 1

         if s1 == 1 && bin(_dist) >= 37
            i1,j1 = idxToCoord(s1, glength(layer))
            i2,j2 = idxToCoord(s2, glength(layer))
         end
      end
   end
   # return counts
   bins = (bins ./ counts) .- avg_sq

   # To synchronize bins with GPU method
   halfdims = floor.(Int32, size(_state)./2)
   maxreach = floor(Int32, sqrt(sum(halfdims.^2)))

   return [1:maxreach;], bins[1:maxreach]
end

function correlationLength(layer, ::Type{Fourier})
   @inline function bin(dist)
      floor(Int32, dist)
   end 

   avg_sq = (sum(state(layer))/nStates(layer))^2

   _state = state(layer)
   ft = fft(_state)
   ft = abs2.(ft)
   
   corrs = real.(ifft(ft))
   bins = zeros(Float32, floor(Int, maxdist(layer)))
   counts = zeros(Float32, floor(Int, maxdist(layer)))

   for j in 1:size(corrs,2)
      for i in 1:size(corrs,2)
         if i == 1 && j == 1
            continue
         end
         _dist = dist(1,1, i, j, layer)
         bins[bin(_dist)] += corrs[i,j]
         counts[bin(_dist)] += 1
      end
   end

   correlations = (bins ./ counts)./(nStates(layer)) .- avg_sq
   return [1:length(correlations);], correlations
end