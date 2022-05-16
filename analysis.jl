# Analysis functions

function corrL(g::IsingGraph, L)
    avgprod = 0
    prodavg1 = 0
    prodavg2 = 0
    Mprod = 0
    M1 = 0
    M2 = 0
    # filter = [ #only do for spin pairs within matrix
    #     let (i1,j1) = idxToCoord(state,g.N), i2 = i1+L, j2 = j1+L
    #         i2 <= g.N && j2 <=g.N
    #     end
    #     for state in 1:g.size
    # ]
    # for state1 in g.state[filter]
    for stateIdx in g.aliveList

        state1 = g.state[stateIdx]
        (i1,j1) = idxToCoord(stateIdx,g.N)
        i2 = i1+L
        j2 = j1+L

        if i2 < g.N 
            statey = g.state[coordToIdx(i2,j1,g.N)]
            prodavg2 += statey   
            avgprod += state1*statey
            Mprod += 1
            M2 +=1
        end
        
        if j2 < g.N
            statex = g.state[coordToIdx(i1,j2,g.N)]
            prodavg2 += statex
            avgprod += state1*statex
            Mprod += 1
            M2 +=1
        end

        if i2 < g.N || j2 < g.N
            prodavg1 += state1 
            M1 += 1
        end
    end

    return avgprod/Mprod-prodavg1*prodavg2/(M1*M2)

end

function corrFunc(g::IsingGraph, plot = true)
    corr::Vector{Float32} = []
    x = [1:(g.N-1);]
    for L in 1:(g.N-1)
        append!(corr,corrL(g,L))
    end

    if plot
        display(pl.plot(x,corr))
    end

    return corr
end

function magnetization(g::IsingGraph,M,M_array)
    avg_window = 50 # Averaging window = Sec * FPS, becomes max length of vector
    append!(M_array,sum(g.state))
    if length(M_array) > avg_window
        deleteat!(M_array,1)
        M[] = sum(M_array)/avg_window 
    end 
end

function tempSweep(g,TIs, M_array, TF = 13, dpoints = 12, sleeptime = 5, equi_wait = 0)
    println("Starting temperature sweep")

    # Data 
    Ts = []
    corrLs = []
    Ms = []

    TRange =  TIs[]:0.5:TF
    println("The sweep will take approximately $(length(TRange)*dpoints*sleeptime ) seconds")
    for T in TRange
        println("Doing temperature $T, gathering $dpoints data points in intervals of $sleeptime seconds")
        println("Approximately $(length(T:0.5:TF)*dpoints*sleeptime) seconds remaining")
        TIs[] = T
        
        for i in 1:dpoints
            append!(Ts, T)
            append!(corrLs,[corrFunc(IsingGraph(g),false)])
            append!(Ms, last(M_array))
        
            sleep(sleeptime)
        end
        
        sleep(equi_wait)
    end

    return (Ts,corrLs,Ms)
end

function dataToDF(tsData, lMax)
    return DataFrame(Temperature = tsData[1], Correlation_Function = [corrL[1:lMax] for corrL in  tsData[2]], Magnetization = tsData[3] )
end

