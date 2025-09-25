using InteractiveIsing
import InteractiveIsing as ii

function isingfunc(dr, c1, c2)
    return 1/dr
end

wg = ii.@WG (dr,c1,c2) -> 1/dr NN=3

g3d = ii.IsingGraph(10,10,10, type = Discrete, periodic = true)
genAdj!(g3d[1], wg) 

wg = @WG (dr,c1,c2) -> isingfunc(dr, c1, c2) NN=3
wg(;dr=1, c1=Coordinate(1,1,1), c2=Coordinate(2,2,2) )
# createProcess(g3d)
# interface(g3d)