function counts_test(state32, counts,l,w)
    i = thread_position_in_grid_1d()

    if i < length(state32)
        for conn_idx in (i+1):length(state32)
            i1, j1 = idxToCoord(i, l)
            i2, j2 = idxToCoord(conn_idx, l)
            disp_i = Float32(latmod(i1 - i2, l/2f0))
            disp_j = Float32(latmod(j1 - j2, w/2f0))
            dist = sqrt(disp_i^2 + disp_j^2)
            
            dist_bin = unsafe_trunc(Int32, dist)

            Metal.@atomic counts[dist_bin] += 1
        end
    end
end

state4 = copy(state(g[4]))
state4Mtl = MtlArray(state4)
l = Int32(size(state4)[1])
w = Int32(size(state4)[2])
mdist = floor(Int32, maxdist(g[4]))

threads = 32
groups = cld(length(countsMtl), threads)
countsMtl = Metal.zeros(Int32, Int64(mdist))
Metal.@sync @metal threads = threads groups = groups counts_test(state4Mtl, countsMtl, l, w)
countsCPU = Array(countsMtl)
sum(countsCPU)

for idx in eachindex(counts_cpu)
    if counts_cpu[idx] != counts_gpu[idx]
        println("Mismatch at $idx")
        println("CPU: $(Int(counts_cpu[idx]))")
        println("GPU: $(Int(counts_gpu[idx]))")
    end
end