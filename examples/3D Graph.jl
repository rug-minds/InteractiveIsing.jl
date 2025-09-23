using InteractiveIsing
import InteractiveIsing as ii

function IsingWG(d)
    dr2 = norm2(d)
    if dr2 == 1
        return 1
    else
        return 0
    end
end

wg = ii.@WG d -> norm2(d) == 1 ? 1 : 0 NN=1

g3d = ii.IsingGraph(10,10,10, type = Discrete)
createProcess(g3d)
genAdj!(g3d[1], wg) 
interface(g3d)