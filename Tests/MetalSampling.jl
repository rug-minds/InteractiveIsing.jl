using Metal

function torusDist(i1,j1,i2,j2, lngth, wdth)::Float32
    dy::Float32 = abs(i2-i1)
    dx::Float32 = abs(j2-j1)

    if dy > .5f0 *lngth
        dy = lngth - dy
    end
    if dx > .5f0 *wdth
        dx = wdth - dx
    end

    return sqrt(dx^2+dy^2)
end

state_copy = copy(state(l1))

state32 = MtlArray(Float32.(state_copy)); corrs2 = MtlArray(zeros(Float32, (floor.(Int32, (size(state_copy)) ./ 2) .+ 1)...)); counts = similar(corrs2); l, w = Int32.(size(state_copy)); threads2_2d = 16,16; groups2_2d = cld.(size(corrs2), threads2_2d)
dists = similar(corrs2)


function corrMetal(state32, corrs, counts, l ,w, ::Type{Periodic})
    #metal idx
    i, j = thread_position_in_grid_2d()
    i_off = Int32(i - 1)
    j_off = Int32(j - 1)

    corr_l, corr_w = size(corrs)

    i_it = i < corr_l ? (1:l) : (iseven(l) ? (1:(corr_l-1)) : (1:(corr_l-1)))
    j_it = j < corr_w ? (1:w) : (iseven(w) ? (1:(corr_w-1)) : (1:(corr_w-1)))

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

    # innerFuncCorrMetal(state32, corrs, counts, l, w, i ,j, i_off, j_off, corr_l, corr_w, i_it, j_it)

    return nothing
end

function getDistancesMetal(dists, func)
    i, j = thread_position_in_grid_2d()

    if i < size(dists,1) && j < size(dists,2)
        dists[i,j] = func(i,j)
    end
    return
end

# function corrMetal(state32, corrs, l ,w, ::Type{Periodic}, ::Type{Odd})
#     #metal idx
#     i, j = thread_position_in_grid_2d()
#     i_off = Int32(i - 1)
#     j_off = Int32(j - 1)

#     corr_l, corr_w = size(corrs)

#     i_it = i < corr_l ? (1:l) : (1:(corr_l-1))
#     j_it = j < corr_w ? (1:w) : (1:(corr_w-1))

#     innerFuncCorrMetal(state32, corrs, counts, l, w, i ,j, i_off, j_off, corr_l, corr_w, i_it, j_it)

#     return nothing
# end


# @metal threads = threads2_2d groups = groups2_2d corrMetal2(state32Matrix, corrs, l, w, Periodic)

function divideCorrs_andSubtractAvg2(corrs,counts, avg2)
    i = thread_position_in_grid_1d()
    @inbounds if i < length(corrs) && counts[i] > 0
        corrs[i] = corrs[i] / counts[i] - avg2
    end
    return
end

function corrlGPU(layer)
    state_copy = copy(state(layer))
    l, w = Int32.(size(state_copy))

    # Metal Arrays
    stateMtl = MtlArray(Float32.(state_copy))
    corrsMtl = MtlArray(zeros(Float32, (floor.(Int32, (size(state_copy)) ./ 2) .+ 1)...)) 
    countsMtl = similar(corrs2) 
    
    threads2_2d = 16,16
    groups2_2d = cld.(size(corrs2), threads2_2d)

    @metal threads = threads2_2d groups = groups2_2d corrlMetal(stateMtl, corrs, counts, len, wid, periodic(layer))

    #convert back to arrays
    corrs = Array(corrs)
    counts = Array(counts)

    filter = counts .> 0
    corrs = corrs[filter]
    counts = counts[filter]

    corrs = corrs ./ counts
    corrs = corrs .- avg2
    bins = (1:largest_dist)[filter]
    return bins, corrs, counts
end

function corrl(state32, corrs)
    l, w = size(state32)

    #metal idx

    @inbounds for i in 1:l
                for j in 1:w
                    for idx in 1:l
                        for jdx in 1:w
                            if idx != i && jdx != j
                                    dist = (torusDist(i,j,idx,jdx, l, w)) ÷ 1
                                    corrs[unsafe_trunc(Int32,dist)] += state32[i,j]*state32[idx,jdx]
                            end
                        end
                    end
                end
    end

end


