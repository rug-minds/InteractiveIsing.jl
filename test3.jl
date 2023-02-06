using BenchmarkTools

function vecvecIdx(vec, idx)
    function vecvecIdxInner(vec, idx, outeridx)
        tmp_idx = idx - length(vec[outeridx])
        if tmp_idx <= 0 
            return vec[outeridx][idx]
        else
            vecvecIdxInner(vec, tmp_idx, outeridx+1)
        end
    end

    vecvecIdxInner(vec, idx, 1)
end

function vecvecIdxLoop(vec, idx)
    outeridx = 1
    while idx > 0
        tmp_idx = idx - length(vec[outeridx])
        if tmp_idx <= 0 
            return vec[outeridx][idx]
        else
            idx = tmp_idx
            outeridx += 1
        end
    end
end

const vec = [rand(3),rand(3),rand(3)]

@btime vecvecIdx(vec,4)
@btime vecvecIdxLoop(vec,4)