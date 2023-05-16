using Metal

# state_copy = copy(state(l1))

# state32 = MtlArray(Float32.(state_copy)); corrs = MtlArray(zeros(Float32, (floor.(Int32, (size(state_copy)) ./ 2) .+ 1)...)); counts = similar(corrs); l, w = Int32.(size(state_copy)); threads2_2d = 16,16; groups2_2d = cld.(size(corrs), threads2_2d)
# dists = similar(corrs)


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

function getDistancesMetal(dists, func)
    i, j = thread_position_in_grid_2d()

    if i < size(dists,1) && j < size(dists,2)
        dists[i,j] = func(i,j)
    end
    return
end

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

    return corr_bins, bin_counts, corrsMtl
end

function corrl(layer)
    state_copy = copy(state(layer))
    l, w = size(state_copy)

    #metal idx

    corrs = zeros(Float32, floor(Int32,dist(1,1, -floor(Int32,-(l/2)),-floor(Int32,-(w/2)), top(layer))))
    counts = zeros(Int64, floor(Int32,dist(1,1, -floor(Int32,-(l/2)),-floor(Int32,-(w/2)), top(layer))))
    avg2 = (sum(state_copy) / (l*w))^2

    @inbounds for j in 1:w
                for i in 1:l
                    for jdx in j:(j+floor(Int64,w/2)-1)
                        for idx in i:(i+floor(Int64,l/2)-1)
                            if idx != i && jdx != j
                                jdx = jdx > w ? jdx - w : jdx
                                idx = idx > l ? idx - l : idx
                                dist_bin = floor(Int32,dist(i,j,idx,jdx, top(layer)))
                                corrs[dist_bin] += state_copy[i,j]*state_copy[idx,jdx]
                                counts[dist_bin] += 1
                            end
                        end
                    end
                end
    end

    corrs = corrs ./ counts
    corrs = corrs .- avg2
    return [1:length(corrs);] , corrs
end


