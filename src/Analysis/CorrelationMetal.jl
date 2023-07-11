correlationLength(layer, type::Type{Mtl}) = corrMtl(layer)

function corrMtl(layer; periodicity = nothing)
    state_copy = copy(state(layer))
    l, w = Int32.(size(state_copy))
    avg2 = (sum(state_copy) / (l*w))^2

    periodicity = periodicity == nothing ? periodic(layer) : periodicity
    if periodicity <: Periodic
        corrs_size = floor.(Int32, (size(state_copy)) ./ 2)
    else
        corrs_size = floor.(Int32, (size(state_copy)))
    end

    # Metal Arrays
    stateMtl = MtlArray(Float32.(state_copy))
    corrsMtl = MtlArray(zeros(Float32, corrs_size...)) 
    countsMtl = similar(corrsMtl) 
    
    threads2_2d = 16,16
    groups2_2d = cld.(size(corrsMtl), threads2_2d)

    Metal.@sync @metal threads = threads2_2d groups = groups2_2d corrMtlKernel(stateMtl, corrsMtl, countsMtl, l, w, periodicity)

    #convert back to array
    avg_corrs_cpu = Array(corrsMtl)
    counts_cpu = Array(countsMtl)

    topology = LayerTopology(top(layer), periodicity)
    max_dist = dist(1,1, size(avg_corrs_cpu)..., topology)
    max_bin = floor(Int32, max_dist)
    
    corr_bins = zeros(Float32, max_bin)
    bin_counts = zeros(Float32, max_bin)

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
export corrMtl

# GPU KERNEL Periodic
function corrMtlKernel(state32, corrs, counts, l ,w, ::Type{Periodic})
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

    return nothing
end

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
