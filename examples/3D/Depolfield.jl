using InteractiveIsing, LoopVectorization, Processes
import InteractiveIsing as II

g = IsingGraph(40,40,40, type = II.Discrete)
 
simulate(g)


function weightfunc(dx,dy,dz)
    prefac = 1
    dr2 = (dx)^2+(dy)^2+dz^2

    if abs(dy) > 0 || abs(dx) > 0
        prefac *= 1
    end
    if dr2 != 1
        return 0
    end
    return prefac
end


wg = @WG "(dx,dy,dz) -> weightfunc(dx,dy,dz)" NN = (1,1,3)

genAdj!(g[1], wg)

setparam!(g[1], :b, 130, false)
quit(g)

# g.hamiltonian = NIsing(g) + DepolField(g)
g.hamiltonian = Ising(g)
g.hamiltonian = setglobalparam(g.hamiltonian, :b)

createProcess(g, KineticMC)
# as = (;getargs(g)..., newstate = SparseVal(-1,1,1))
# dh = II.deltaH(g.hamiltonian)








