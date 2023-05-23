""" 
Takes a random number of pairs for every length to calculate correlation length function
This only works without defects.
"""

# Should all return x, y, e.g. length_bins and correlation
abstract type SamplingAlgorithm end

struct Mtl <: SamplingAlgorithm end

correlationLength(layer, type::Type{Mtl} = Mtl) = corrlGPU(layer)
export correlationLength

function corrlGPU(layer)
    state_copy = copy(state(layer))
    l, w = Int32.(size(state_copy))
    avg2 = (sum(state_copy) / (l*w))^2

    # Metal Arrays
    stateMtl = MtlArray(Float32.(state_copy))
    corrsMtl = MtlArray(zeros(Float32, (floor.(Int32, (size(state_copy)) ./ 2))...)) 
    countsMtl = similar(corrsMtl) 
    
    threads2_2d = 16,16
    groups2_2d = cld.(size(corrsMtl), threads2_2d)

    Metal.@sync @metal threads = threads2_2d groups = groups2_2d corrMetal(stateMtl, corrsMtl, countsMtl, l, w, periodic(layer))

    #convert back to array
    avg_corrs_cpu = Array(corrsMtl)
    counts_cpu = Array(countsMtl)

    topology = top(layer)
     
    corr_bins = zeros(Float32, floor(Int32,dist(1,1, size(avg_corrs_cpu,1),size(avg_corrs_cpu,2), topology)))
    bin_counts = zeros(Float32, floor(Int32,dist(1,1, size(avg_corrs_cpu,1),size(avg_corrs_cpu,2), topology)))

    for j in 1:size(avg_corrs_cpu,2)
        for i in 1:size(avg_corrs_cpu,1)
            if i == 1 && j == 1
                continue
            end
            # println("i: $i, j: $j")
            # println("dist: $(dist(1,1, i,j, topology))")
            distance_bin = floor(Int32, dist(1,1, i,j, topology))
            corr_bins[distance_bin] += avg_corrs_cpu[i,j]
            bin_counts[distance_bin] += counts_cpu[i,j]
        end
    end

    filter = bin_counts .> 0
    corr_bins = corr_bins[filter]
    bin_counts = bin_counts[filter]

    corr_bins = (corr_bins ./ bin_counts) .- avg2

    return [1:length(corr_bins);], corr_bins
end
# precompile(correlationLength, IsingLayer)

# GPU KERNEL Periodic
function corrMetal(state32, corrs, counts, l ,w, ::Type{Periodic})
    #metal idx
    i, j = thread_position_in_grid_2d()
    i_off = Int32(i - 1)
    j_off = Int32(j - 1)

    corr_l, corr_w = size(corrs)

    i_it = 1:l
    j_it = 1:w

    count = 0
    if i_off < corr_l && j_off < corr_w
        for j1 in j_it
            for i1 in i_it
                i2::Int32 = i1 + i_off > l ? i1 + i_off - l : i1 + i_off
                j2::Int32 = j1 + j_off > w ? j1 + j_off - w : j1 + j_off
                corrs[i,j] += state32[i1,j1]*state32[i2,j2]
                count +=1
            end
        end
        counts[i,j] = count
    end
    return

    return nothing
end

# Old

function sampleCorrPeriodic(layer; Lstep::Float16 = Float16(.5), lStart::Integer = Int32(1), lEnd::Integer = Int16(256), precision_fac = 5, npairs::Integer = Int32(precision_fac*10000) )
    
    layer_copy = deepcopy(layer)
    alives = aliveList(layer_copy)

    function sigToIJ(sig, L)
        return (L*cos(sig),L*sin(sig))
    end

    function sampleIdx2(idx1,L,rtheta, layer)
        # turn idx into coordinates
        i,j = idxToCoord(idx1,glength(layer))
        # Turn angle and length into relative coordinates
        dij = Int32.(round.(sigToIJ(rtheta,L))) 
        # Turn old coordinates plus relative coordinates into and index again
        idx2 = coordToIdx(latmod(i+dij[1], glength(layer)),latmod(j+dij[2], gwidth(layer)), gwidth(layer))

        # Old code for square lattice
        # idx2 = coordToIdx(latmod.((ij.+dij),g.N),g.N)
        return idx2
    end
    avgsum = (sum(state(layer_copy))/nStates(layer_copy))^2

    lVec = [lStart:Lstep:lEnd;]
    corrVec = Vector{Float32}(undef,length(lVec))

    # Sample all startidx to be used
    # Slight bit faster to do it this way than to sample it every time in the loop
    idx1s = rand(alives,length(lVec)*npairs)
    # Index of above vector
    idx1idx = 1
    # Iterate over all lengths to be checked
    Threads.@threads for (lidx,L) in collect(enumerate(lVec))
    # for (lidx,L) in enumerate(lVec)
 
        sumprod = 0 #Track the sum of products sig_i*sig_j
        for _ in 1:npairs
            idx1 = idx1s[idx1idx]
            
            rtheta = 2*pi*rand()
            idx2 = sampleIdx2(idx1,L,rtheta, layer)
            sumprod += state(layer_copy)[idx1]*state(layer_copy)[idx2]
            idx1idx += 1 # Sample next random startidx
        end
        # println((sum(g.state)/g.N)^2)
        # println(avgsum1*avgsum2/(npairs^2))
        corrVec[lidx] = sumprod/npairs - avgsum
    end

    return (lVec,corrVec)
end
export sampleCorrPeriodic

# Sample correlation length function when there are defects.
function sampleCorrPeriodicDefects(layer::IsingLayer; lend = -floor(-sqrt(2)*max(gwidth(layer),glength(layer))/2), binsize = .5, precision_fac = 1, npairs::Integer = Int64(round(lend/binsize * precision_fac*40000)), sig = 1000, periodic = true)

    g = deepcopy(layer)

    function torusDist(i1,j1,i2,j2, g)
        dy = abs(i2-i1)
        dx = abs(j2-j1)

        if dy > .5*glength(g)
            dy = glength(g) - dy
        end
        if dx > .5*gwidth(g)
            dx = gwidth(g) - dx
        end

        return sqrt(dx^2+dy^2)
    end
    alives = aliveList(g)

    if length(alives) <= 2
        error("Too little alive spins to do analysis")
        return
    end

    idxs1 = rand(alives,npairs)
    idxs2 = rand(alives,npairs)
    lbins = zeros(length(1:binsize:lend))
    lVec = [1:binsize:lend;]
    corrbins = zeros(length(lVec))
    prodavg = sum(state(g)[alives])/length(alives)

    for sample in 1:npairs
        idx1 = idxs1[sample]
        idx2 = idxs2[sample]
        
        while idx1 == idx2
            idx2 = rand(alives)
        end

        (i1,j1) = idxToCoord(idx1,glength(g))
        (i2,j2) = idxToCoord(idx2,glength(g))
        if periodic
            l = torusDist(i1,j1,i2,j2,g)
        else
            l = sqrt((i1-i2)^2+(j1-j2)^2)
        end
        
        # println("1 $idx1, 2 $idx2")
        # println("1 $((i1,j1)), 2 $((i2,j2)) ")
        # println("L $l")

        binidx = Int32(floor((l-1)/binsize)+1)
        
        lbins[binidx] += 1
        corrbins[binidx] += state(g)[idx1]*state(g)[idx2]

    end

    remaining_idxs = []
    for (startidx,pairs_sampled) in enumerate(lbins)
        if pairs_sampled >= sig
            append!(remaining_idxs, startidx)
        end
    end

    corrVec = (corrbins[remaining_idxs] ./ lbins[remaining_idxs]) .- prodavg
    lVec = lVec[remaining_idxs]

    return (lVec,corrVec)
    
end