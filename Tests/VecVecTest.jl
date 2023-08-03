struct VecVec
    vv::Vector{Vector}
end

VecVec(el::T) where T = VecVec([T[el]])
VecVec(els::Vector) = VecVec([els])


#Index as normal matrix
function Base.getindex(vv::VecVec, idx::Integer)
    for i in 1:length(vv.vv)
        if idx <= length(vv.vv[i])
            return vv.vv[i][idx]
        else
            idx -= length(vv.vv[i])
        end
    end
end

#Index as matrix
function Base.getindex(vv::VecVec, idx1::Integer, idx2::Integer)
    return vv.vv[idx1][idx2]
end

# Push to end of ith Vector
function Base.push!(vv::VecVec, idx::Integer, val)
    push!(vv.vv[idx], val)
end

# vecvec iterator
function Base.iterate(vv::VecVec, state = (1,1))
    if state[1] > length(vv.vv)
        return nothing
    elseif state[2] > length(vv.vv[state[1]])
        return iterate(vv, (state[1] + 1, 1))
    else
        return (vv.vv[state[1]][state[2]], (state[1], state[2] + 1))
    end
end

# test
vv = VecVec([[1:10;], [11:20;], [21:30;]])
