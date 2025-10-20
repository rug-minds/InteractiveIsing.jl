using InteractiveIsing, Plots, GLMakie, FileIO
using InteractiveIsing.Processes
import InteractiveIsing as II

xL = 100  # Length in the x-dimension
yL = 100  # Length in the y-dimension
zL = 10   # Length in the z-dimension
g = IsingGraph(xL, yL, zL, stype = Continuous(), periodic = (:x,:y))
# Visual marker size (tune for clarity vs performance)
II.makie_markersize[] = 0.3

# Launch interactive visualization (idle until createProcess(...) later)
interface(g)

temp(g,10)
g.hamiltonian = Ising(g) + DepolField(g, c=60000, left_layers=1, right_layers=1)
g.hamiltonian = sethomogenousparam(g.hamiltonian, :b)

homogeneousself!(g,2)

# wg1 = @WG weightfunc_xy_antiferro NN = (2,2,2)
wg1 = @WG weightfunc1 NN = (1,1,1)
genAdj!(g[1], wg1)

fullsweep = xL*yL*zL
Time_fctr = 0.2
SpeedRate = Int(Time_fctr*fullsweep)

### risepoint and Amptitude are factors from pulse
risepoint=500
Amptitude =10
# risepoint = round(Int, Amptitude/0.01)

### Run with TrianlePulseA
###  /\
### /  \    _____
###     \  /
###      \/

PulseN = 2
Pulsetime = (PulseN * 4 + 2) * risepoint * SpeedRate

compalgo = CompositeAlgorithm((Metropolis, TrianglePulseA), (1, SpeedRate))
createProcess(g, compalgo, lifetime =Pulsetime, amp = Amptitude, numpulses = PulseN, rise_point=risepoint)
### estimate time
est_remaining(process(g))

# Wait until it is done
args = process(g) |> fetch # If you want to close ctr+c
# args = process(g) |> getargs
# EnergyG= args.all_Es;
voltage= args.x
Pr= args.y;

# w1=newmakie(lines, Pr, EnergyG)
# inlineplot() do 
#     lines(voltage, Pr)
# end
w2=newmakie(lines, voltage, Pr)
w3=newmakie(lines,Pr)