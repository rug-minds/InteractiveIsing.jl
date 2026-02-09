using SparseArrays, Random
state = rand([-1,1], 1000)
w = sprand(1000, 1000, 0.001)

mutable struct Randcounter{R}
    r::R
    count::Int
end

Base.rand(rc::Randcounter) = (rc.count += 1; rand(rc.r))

rc = Randcounter(MersenneTwister(), 0)

function test(state,w, rc, ecounter)
    for _ in 1:1000
        #pick i
        i = rand(1:length(state))
        newstate = -state[i]
        oldstate = state[i]
        ΔE = (oldstate-newstate)*((w*state)[i])
        efac = exp(-ΔE)
        # randnum = rand(rc)
        if ΔE < 0
            ecounter[] += 1
        end
        if (ΔE < 0 ||  rand(rc) < efac)
            state[i] = newstate
        end
    end
end
ecounter = Ref(0)
test(state, w, rc, ecounter)
