""" 
Takes a random number of pairs for every length to calculate correlation length function
This only works without defects.
"""
rthetas = 2*pi .* rand(10^7) # Saves random angles to save computation time
function sampleCorrPeriodic(layer, Lstep::Float16 = Float16(.5), lStart::Integer = Int32(1), lEnd::Integer = Int16(256), precision_fac = 5, npairs::Integer = Int32(precision_fac*10000) )
    
    g = deepcopy(layer)
    alives = aliveList(g)

    function sigToIJ(sig, L)
        return (L*cos(sig),L*sin(sig))
    end

    function sampleIdx2(idx1,L,rtheta)
        # turn idx into coordinates
        i,j = idxToCoord(idx1,glength(g))
        # Turn angle and length into relative coordinates
        dij = Int32.(round.(sigToIJ(rtheta,L))) 
        # Turn old coordinates plus relative coordinates into and index again
        idx2 = coordToIdx(latmod(i+dij[1], glength(g)),latmod(j+dij[2], gwidth(g)), gwidth(g))

        # Old code for square lattice
        # idx2 = coordToIdx(latmod.((ij.+dij),g.N),g.N)
        return idx2
    end

    theta_i = rand([1:length(rthetas);])

    avgsum = (sum(state(g))/nStates(g))^2

    lVec = [lStart:Lstep:lEnd;]
    corrVec = Vector{Float32}(undef,length(lVec))

    # Sample all startidx to be used
    # Slight bit faster to do it this way than to sample it every time in the loop
    idx1s = rand(alives,length(lVec)*npairs)
    # Index of above vector
    idx1idx = 1
    # Iterate over all lengths to be checked
    for (lidx,L) in enumerate(lVec)
 
        sumprod = 0 #Track the sum of products sig_i*sig_j
        for _ in 1:npairs
            idx1 = idx1s[idx1idx]
            rtheta = rthetas[(theta_i -1) % length(rthetas)+1]
            idx2 = sampleIdx2(idx1,L,rtheta)
            sumprod += state(g)[idx1]*state(g)[idx2]
            theta_i += 1 # Sample next random angle
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
function sampleCorrPeriodicDefects(layer::IsingLayer, lend = -floor(-sqrt(2)*max(gwidth(layer),glength(layer))/2), binsize = .5, precision_fac = 1, npairs::Integer = Int64(round(lend/binsize * precision_fac*40000)); sig = 1000, periodic = true)
    
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