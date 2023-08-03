function register(g)
    fstate = state(g)
    fadj = adj(g)
    fhtype = htype(g)
    fstype = stype(g)

    testhtype(g,fstate ,fadj, fhtype)
    teststype(g,fstate ,fadj, fstype)
end
const repeats = 10^9
function testhtype(g, fstate, fadj, fhtype)
    ti = time()
    cum_e = 0.f0
    for _ in 1:repeats
        cum_e += getEFactor(g, fstate, fadj, 1, fhtype)
    end
    println("Time: ", time() - ti)
    return cum_e
end

function teststype(g, fstate, fadj, fstype)
    ti = time()
    cum_e = 0.f0
    for _ in 1:repeats
        cum_e += getEFactor(g, fstate, fadj, 1, fstype)
    end
    println("Time: ", time() - ti)
    return cum_e
end

register(g)