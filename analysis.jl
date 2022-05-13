# Analysis functions

function corrL(g::IsingGraph, L)
    avgprod = 0
    prodavg1 = 0
    prodavg2 = 0
    M = 2*(g.N-L)*g.N
    # filter = [ #only do for spin pairs within matrix
    #     let (i1,j1) = idxToCoord(state,g.N), i2 = i1+L, j2 = j1+L
    #         i2 <= g.N && j2 <=g.N
    #     end
    #     for state in 1:g.size
    # ]
    # for state1 in g.state[filter]
    for state1 in g.state
        (i1,j1) = idxToCoord(state1,g.N)
        i2 = i1+L
        j2 = j1+L
        state2 = g.state[coordToIdx(i2,j2,g.N)]
        avgprod += state1*state2
        prodavg1 += state1
        prodavg2 += state2
    end

    return avgprod/M-prodavg1*prodavg2/(M^2)

end

function corrLengthFunction(g::IsingGraph)
    corr::Vector{Float32} = []
    x = [1:(g.N-1);]
    for L in 1:(g.N-1)
        append!(corr,corrL(g,L))
    end
    # plot(x,corr)
    pl.plot(x,corr)
end

function magnetization(g::IsingGraph,M,M_vec)
    avg_window = 5*30 # Averaging window = Sec * FPS, becomes max length of vector
    append!(M_vec,sum(g.state))
    if length(M_vec) > avg_window
        deleteat!(M_vec,1)
        M[] = sum(M_vec)/avg_window 
    end 
    
end