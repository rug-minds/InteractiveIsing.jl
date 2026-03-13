using InteractiveIsing, JET, BenchmarkTools
using InteractiveIsing.Processes
import InteractiveIsing as ii

function isingfunc(dr, c1, c2)
    return 1/dr
end

wg = @WG (dr,c1,c2) -> isingfunc(dr, c1, c2) NN=(3,3,2)

g = ii.IsingGraph(100,100,10, 
        Continuous(), 
        wg, 
        LatticeConstants(1.0, 1.0, 20.),
        StateSet(-1.5f0, 1.5f0),
        Ising(c = ConstVal(1)) + 
            Clamping(1f0)+ Quartic(c = ConstVal(1.0), ) + 
            Sextic(c = ConstVal(1.0), localpotential = StateLike(OffsetArray, 0)),
        periodic = (:x,:y))

interface(g)
createProcess(g, lifetime = Processes.Until(x -> x == 0, Var(g.default_algorithm, :T)))



# pause(g)
# c = context(g)
# reg = getregistry(c)
# cview = view(c, reg[1][1])
# step!(InteractiveIsing.Metropolis(), cview)
# @code_warntype step!(InteractiveIsing.Metropolis(), cview)

