correlationLength(layer, type::Type{Mtl}; kwargs...) = corrMtl(layer; kwargs...)

function corrMtl(layer; periodicity = nothing)
    # Will hang when run from other thread
    GC.enable(false)

    state_copy = copy(state(layer))
    l, w = Int32.(size(state_copy))
    avg2 = (sum(state_copy) / (nStates(layer)))^2

    periodicity = periodicity == nothing ? periodic(layer) : periodicity
    if periodicity <: Periodic
        corrs_size = floor.(Int32, (size(state_copy)) ./ 2) .+  1
    else
        corrs_size = size(state_copy)
    end

    Metal.@sync begin
        # Metal Arrays
        
        # Copy the state to the GPU
        stateMtl = MtlArray(state_copy)


        # Matrix to store correlation for every displacement Vector
        # 3 dimension is for flip of the x displacement
        corrsMtl = Metal.zeros(Float32, Int64.(corrs_size)...,2)

        
        # Matrix to store the number of samples for every displacement Vector
        countsMtl = Metal.zeros(Float32, Int64.(corrs_size)...,2)
        
        
        threads3d = 16,16,2
        groups3d = cld.(size(corrsMtl), threads3d)

        # println("size: $(size(corrsMtl))")
        # println("threads: $threads2_2d, groups: $groups2_2d")

        @metal threads = threads3d groups = groups3d corrMtlKernel(stateMtl, corrsMtl, countsMtl, l, w, periodicity)

        #convert back to array
        # This takes a lot of time?
        # Because async?
        corrsCPU = Array(corrsMtl)
        countsCPU = Array(countsMtl)
    end

    # Get the topology for the graph
    topology = LayerTopology(top(layer), periodicity)

    # What is the maximum disrtance that is sampled
    max_dist = dist(1,1, size(corrsCPU)[1:2]..., topology)

    # From the maximum sampled distance get the maximum bin
    max_bin = floor(Int32, max_dist)
    
    # Initialize the bins
    corr_bins = zeros(Float32, max_bin)
    bin_counts = zeros(Float32, max_bin)

   
    # z=1 loop
    for j in 1:size(corrsCPU,2)
        for i in 1:size(corrsCPU,1)
            if i == 1 && j == 1
                continue
            end

            distance_bin = floor(Int32, dist(1,1, i,j, topology))
            corr_bins[distance_bin] += corrsCPU[i,j,1]
            bin_counts[distance_bin] += countsCPU[i,j,1]
        end
    end

    # z=2 loop
    j_it = iseven(size(state_copy,2)) ?  (2:(size(corrsCPU,2)-1))   : (2:size(corrsCPU,2))
    i_it = iseven(size(state_copy,1)) ?  (2:(size(corrsCPU,1)-1))   : (2:size(corrsCPU,1))

    for j in j_it
        for i in i_it
            distance_bin = floor(Int32, dist(1,1, i,j, topology))    

            corr_bins[distance_bin] += corrsCPU[i,j,2]
            bin_counts[distance_bin] += countsCPU[i,j,2]
        end
    end

    corr_bins = (corr_bins ./ bin_counts) .- avg2
    # corr_bins = (corr_bins ./ bin_counts)

    GC.enable(true)
    return [1:length(corr_bins);], corr_bins
end
export corrMtl


"""
Kernel for computation of the correlation length function

The idea is to assign a displacement vector to each thread and then move these displacement vectors
over the lattice. This way the correlation for each displacement vector can be computed in parallel
and later aggregated into distance bins by the cpu in a sequential part.

For even lattices, there is a problem.
A priori, to get all the correlation terms σ_0 σ_k, we need a displacement vector from the first spin
to every other spin. Thus we have a set of vectors that point into the lower right quadrant of the plane.
However, under translations of the starting points, if we use the full set of
displacement vectors, on a periodic grid, if we go from σ_k to σ_l, there will be another displacement vector
that gets back to σ_k, causing us to overcount correlations. We will call these vectors dual. In our set, if there is
a dual vector, we just need to choose one of those two and not include the other. There are vectors that are self-dual.
This means that if we include them, we overcount, but if we don't include them, we undercount. These vectors always
are purely in the x or y direction with length l/2 or w/2 respectively or purely diagonal with both dimensions being l/2 and w/2.

To deal with self dual vectors, we need to change the iteration range from all the points in the grid, to a half rectangle.
"""
function corrMtlKernel(state32, corrs, counts, l ,w, ::Type{Periodic})
    #metal idx
    i, j, z = thread_position_in_grid_3d()

    # Assign a displacement vector to each thread
    # Displacement vectors start at (0,0)
    i_off = Int32(i - 1)

    if z == 1
        j_off = Int32(j - 1)
    else
        j_off = Int32(1 - j)
    end


    maxl = unsafe_trunc(Int32, Float32(l)/2f0)
    maxw = unsafe_trunc(Int32, Float32(w)/2f0)

    # Iteration ranges
    i_it = 1:l
    j_it = 1:w
    
    corr_l, corr_w = size(corrs)


    if i_off < corr_l && j_off < corr_w
        # If the vector is self dual, reduce the iteration ranges
        # Otherwise just iterator over every spin
        if (iseven(l) && i_off == maxl) || (iseven(w) && j_off == maxw) || (iseven(l) && iseven(w) && i_off == maxl && j_off == maxw)
            if j_off == maxw && i_off == 0
                j_it = 1:maxw
            end

            if i_off == maxl && j_off == 0
                i_it = 1:maxl
            end

            if i_off == maxl && j_off == maxw
                j_it = 1:maxw
            end
            
        end

        count = 0
        for j1 in j_it
            for i1 in i_it
                i2 = latmod(i1 + i_off, l)
                j2 = latmod(j1 + j_off, w)
                corrs[i,j,z] += state32[i1,j1]*state32[i2,j2]
                count +=1
            end
        end

        counts[i,j,z] = count

    end

    return nothing
end
export corrMtlKernel

# GPU KERNEL NonPeriodic
function corrMtlKernel(state32, corrs, counts, l ,w, ::Type{NonPeriodic})
    i, j = thread_position_in_grid_2d()
    i_off = Int32(i - 1)
    j_off = Int32(j - 1)

    corr_l, corr_w = size(corrs)

    i_it = 1:(l-i_off)
    j_it = 1:(w-j_off)

    count = 0
    if i_off < corr_l && j_off < corr_w
        for j1 in j_it
            for i1 in i_it
                
                i2::Int32 = i1 + i_off
                j2::Int32 = j1 + j_off

                corrs[i,j] += state32[i1,j1]*state32[i2,j2]
                count +=1
            end
        end
        counts[i,j] = count
    end

    return nothing
end