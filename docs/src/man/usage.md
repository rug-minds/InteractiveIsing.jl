# Before Using

This package aims to offer interactive simulations that run at the maximal speed Julia has to offer. Due to the nature of interactive simulations, we need multithreading. Therefore, starting up julia as normal (which defaults to 1 thread), will cause freezing when trying to use this package.

To start julia up, either start it from a terminal with multiple threads (at least 8 should be supported on many modern CPU's)
```
julia --threads 8
```

or add an environment variable (this depends on the operating system)
```
JULIA_NUM_THREADS=8
```

When using through VSCode, add the following setting:

```
"julia.NumThreads" : "8"
```

If you experience hanging still on certain operations, start with
```
"julia.NumThreads" : "7,1"
```

# General Usage


This package can be used to simulate 2-, 2.5- and 3D Ising models (any model with 1D state and where the couplings are at most linear.) with different algorithms and energy functions.
The simulations are segmented in different layers which can be 2D or 3D (at this moment there's only support for rectangular and cuboid layers).

The base datastructure for every simulation is the IsingGraph, which holds all the data neccesary for the simulation to run, such as the state, the adjacency list, other parameters (like the magnetic field or other parameters in the energy function) and metadata for the layers, among other things.

The layers are an abstraction on top of the basic IsingGraph, and are the structures the user will mainly interact. Any layer in itself can be interpreted as an Ising Model with it's own properties. Currently the library supports layers that have continuous or discrete states. For both of these types a "state set" can be defined. For continuous layers the stateset may consist of two numbers which are interpreted as the end points of a closed interval. The states of any spin may then take on values from anywhere in this interval. For discrete states, the stateset may consist of any amount of numbers, and the spins in the system may take any of the values in the set.

We can create an empty graph by calling the constructor for the IsingGraph

```
# Construct Empty Graph
g = IsingGraph()
```

In this guide we will add everything step by step, but there are additional constructors to instantiate a graph with layers in place already. See [IsingGraphs](@ref) for more information.

## Adding Layers

We can add layers one by one in the following way
```
#addLayer!(len, width, height = nothing; set, type)
#Add a 2D layer
addLayer!(g, 250,250, set = (-1,1), type = Continuous)
...
```
The exlamation mark denotes that this function is altering data in the struct it is being passed (in this case the graph g), as per general convention.
The argument height is optional, and if not the function produces a 2D layer. The arguments behind the semicolon ';' are keyword arguments that have to be explicitly given by writing the name followed by an equals sign like shown.

By default, these layers have no connections internally or to other layers.

## Accessing the layers

Now that our IsingGraph contains multiple layers, when we want to interact with the graph, we typically do so through these layers directly instead of with the graph itself (since the graph is a more abstract representation). We can access the layers of a graph through vector indexing of the graph. E.g., we can access the n-th layer of a graph as follows

```
layer_n = g[n]
```

where n is some integer.

Most functions that work on an IsingGraph also work on an IsingLayer directly. However, if we refer to a spin in the graph or a spin in the layer the index we use to refer to a spin typically is different.

## Indexing the spins
The final state that is updated is held in a long vector stored in the IsingGraph. The layers, then, hold references to a part of this long vector and are interpreted as 2D or 3D structures. As an example we can have an IsingGraph with 3 2D layers,

$g \equiv [s_1,...,s_{N_1},s_{N_1 + 1}, ..., s_{N_2}, ... ]$

where, for example, spins $s_{N_1 + 1}, ..., s_{N_2}$ would all belong to the second layer. This second layer has a 2D dimension of $L_2*W_2 = N_2 - (N_1 + 1)$ where we define the length $L$ to be the vertical dimension and the width $W$ the horizontal dimension. Thus we may interpret this part of the vector in the graph, indexed by a single index, as a 2D matrix in the layer, which we can access by giving two coordinates i,j. The coordinates correspond the the index in the following way (known in CS as column-major order): $index = \left( ( (i-1) mod L) + 1 + L*j \right)$. In words, when we iterate through values of $i$, we cycle through numbers $1$ to $50$, and for every value of $j$ we add $50$ to that to get the index.

See [Indexing](@ref) for supporting functions.

## Generating the weights

The connections for a layer and between layers can be generated using a [WeightGenerator](@ref). A weightgenerator must be constructed using the macro `@WG`. A weightgenerator is a struct that can be used to get a weight based on an arbitrary function of any combination of the following arguments: `[:dr, :x, :y, :z, :dx, :dy, :dz]`, where dr is the relative distance between two spins, x, y and z are the midpoints between two spins that a weight is connecting (counted from the left top) and dx dy and dz are the separate one dimensional, relative distances between two spins. A weight generator expects a string containing a julia anonymous function in the following way

```
# Create the weightgenerator
wg_Ising = @WG "dr == 1 ? 1 : 0" NN = 1 # Ising Connections, taking into account only connections 1 spin apart in any direction

wg = @WG "(dr) -> 1/dr^2" NN = (2,3) #Arbitrary function with two nearest neighbors in x direction, 3 in y
```

A weightgenerator can also include additive or multiplicative noise based on arbitrary distributions. See the [WeightGenerator](@ref) page for more information on the usage.

To now generate and set the connections within layer `i` we use the function
```
genAdj!(g[i], wg)
```
## Setting Defects

When the simulation is started, the simulation loop is aware at the start which spins may be updated. Generally this will include every spin, but we can let the simulation know it should only loop over a collection of spins, effectively freezing the others. This will cause a small performance hit, because choosing a spin from a continuous range is less expensive than choosing one from a list, but generally the effect should be small and mostly independent on the size of the list.

We can set defect with the following function

```
setDefects!(g::AbstractIsingGraph, bool, idxs)
```

where an AbstractIsingGraph can be either an IsingGraph or an IsingLayer. I.e. spins may be set using the list index of the underlying graph, or using the list index of a specific layer.

If `bool == true`, then we indicate the spins at the given indexes should be considered as defect (i.e. non-updating), whereas `bool == false` indicates the spins may be updated again.

In some sense the list of defects may be considered as metadata, meaning that it's just information that may be used by an algorithm, but doesn't neccesarily have to be used. In other words, the specific algorithm being used may still update the defect spins, or for example use the list of defects to apply a different type of update to those spins. It all depends on how the algorithm is implemented to use this data.

We may directly set spins to a value and set them to be defect or conversely updating using

```
setSpins!(g::AbstractIsingGraph, idx::Integer, brush::Real, clamp::Bool = false)
```

For more information see [Defects](@ref)

## Starting the simulation

When the graph is updated using the built in simulation, it refers to a simulation struct that's held in the module. A user typically doesn't need to interact with this struct directly. It holds data for the windows that are open, the multiple threads being used for simulation loops on a single graph or multiple different ones, etc.

Now that we have set up the graph, we typically start a simple simulation with the following function

```
simulate(g::IsingGraph, gui = true)
```

where the optional keyword argument `gui` may be given to opt out of starting the graphical user interface (this may be useful if custom display options need to be used).

The interface, and handling of julia commands all run on a separate thread from the simulation loop itself. Any updating algorithm is typically run on a separate thread on the computer, which updates the graph asynchronously. I.e. in this setup a user will not be sure exactly on which loop interaction will happen.

Any of the algorithms that are deployed, depend on a so called loop function. At this moment the only loop function is the mainLoop function, which is just a loop that runs indefinitely. Support for a loop function that loops for a predefined number of iterations after which it will run given analysis functions, will be added. For more information on how the loops work, see [Loops](@ref)


## Interacting with the simulation loop

Once we have a simulation running, we might want to pause, restart it with different parameters or a different algorithm or stop it completely for a while.

Generally we can pause using the function

```
pause(g)
```

after which it can be resumed with

```
unpause(g)
```

We can also completely quit the simulation though this is generally not recommended

```
quit(g)
``` 

## Interacting with parameters

Hamiltonians, and sometimes the algorithms themselves, might have parameters that either have a physical interpretation or some technical interpretation in terms of the simulation itself. Moreover, these parameters might be accesible to the user to change during runtime, like the magnetic field on the spins, or a clamping factor for equilibrium propagation.

Parameters will be either of scalar of vector value. In the case of a vector value, every index of the vector will correspond one-to-one to a spin.

Since the parameters that are in the graph will depend on the simulation loop and hamiltonian, all of which can be changed during runtime, they are not put in place when the graph is created (this might change), but rather when the simulation is started. Every time a simulation is started or restarted, the simulation loop will check if all the neccesary parameters are in place.

These parameters may all be accesed through the accessor function `params(g)`. This will print out a list of parameters, which are referred to by a julia symbol, along with an explanation, whether it should be used by a simulation loop, and what its current value is (it may be scalar or vector valued).

### Default values

Since accessing memory is generally costly in computation time, acessing many parameters will generally cause a simulation loop to slow down. In other words, even if for some simulation a magnetic field is never considered, having the option to set a magnetic field would cause a slow down regardless of wether it is used. This is why these parameters are wrapped in a `ParamVal` type, which holds extra information, i.e. what a default global value is for the parameter in question, and wether it should be considered in the simulation.

Using Julias metaprogramming capabilities simulation loops may substitute the global value for a parameter into the loop itself instead of acessing memory, in the case where a parameter is noted as not being active. This has a downside, in that the simulation needs to be paused and recompiled when the active status of a parameter is changed, or the global value it has. In other words, we made a trade-off between simulation speed and startup/restart time, in favor of simulation speed. In the most optimal case, this also causes extra overhead when writing new simulation loops or Hamiltonians, since metaprogramming is needed. However, a simulation loop may opt out of using the active status of a parameter altogether, so that prototyping new algorithms still has the same amount of overhead.

### Setting and getting parameters

Since the parameters contain information in their types for the simulation to use, which requires recompilation of the loop if changed, it is generally not advised to directly change the data in the parameters itself. Instead we provide the function `setparam!`, which automatically handles recompilation of running loops.

```
setparam!(g::AbstractIsingGraph, param::Symbol, val::T, active::Bool = nothing)
```

where `T` either is either of the same type of the value of the parameter itself, or in the case that the `ParamVal` holds a vector value, it may be of the element type of that vector, in which case the whole vector is globally set to the same value. In the latter case the user may also opt to deactivate the param and set a global value. This is only advised if the value is not changed often, since every change then causes recompilation of the loop. `param` here is a julia symbol that refers to the parameter that the user wants to change. `g` can be a graph or a layer, which only matters for vector values. In the layer case, we may give a scalar, a vector or a matrix as val. If a scalar is given it sets a every value corresponding to a spin in that layer to the same value when a scalar is given. For a vector, the function expects a vector with length equal to the number of spins in the layer. Lastly, for the matrix, it must be of the same size of the layer.

Finally if the optional argument `active` is given it will change wether value of that parameter should be accessed during runtime in the simulation loop. If this argument is not provided, it will keep the previous active status.

To immediately get the parameter of corresponding to a symbol, we can use

```
getparam(g::AbstractIsingGraph, param::Symbol)
```
See [Parameters](@ref) for more information.

## Algorithms and Hamiltonians

Hamiltonians in this package are an abstract description of an energy function in terms of Julia symbols. They don't have any inherent code, and are only meant to symbolize the existense of that Hamiltonian in this package. The reason for this is the following: we want to support many different algorithms. These algorithms may require different quantities/functions derived from the Hamiltonian (such as a difference when changing the state of one unit or a derivative). Hardcoding the code for a Hamiltonian would not be efficient, and automating rewriting, or deriving quantities would still not give optimal performance, apart from being nigh impossible to implement for every algorithm.

Instead we have a system where the writer of an algorithm needs to implement a custom piece of code that implements the Hamiltonian for the specific algorithm. It is thus up to the writer of an algorithm to implement Hamiltonians.

For the user, different algorithms and Hamiltonians can be chosen. Moreover, algorithms can also implement composite Hamiltonians ($H_{tot} = H_1 + H_2 + ...$). 

We can set a default algorithm to run when a new simulation is started (which can be manually overwritten when starting the simulation as well)

```
default_algorithm(g, LayeredMetropolis)
```

A new Hamiltonian can be set the following way

```
hamiltonian(g,Ising)
```

A composite Hamiltonian can be constructed from

```
CompositeHamiltonian(Ising,Clamping)
```
### Algorithms
The following algorithms are currently Implemented
* `Metropolis`: The normal metropolis algorithm. This algorithm treats every layer to be of the same type as layer 1.
* `LayeredMetropolis`: An algorithm that implements a different algorithm for every layer based on the layer type. Due to metaprogramming it just reduces to the normal metropolis when all layers are of a single type.
*  Soon: `Langevin`

### Hamiltonians
Currently the following Hamiltonians are implemented

* `Ising`: The Ising Hamiltonian $H = - \sum_{ij} \sigma_i \sigma_j -\sum_i b_i \sigma_i$
    where $b$ is the magnetic field at every spin
* `Clamping`: A Clamping Hamiltonian for Equilibrium Propagation $H = β/2 *(s_i - y_i)^2$
    where $y_i$ is a target value for the n-th spin
* `GaussianBernoulli`: The Gaussian Bernoulli Hamiltonian often used with RBM's 


## Processes

At the heart of the software are threaded loops that run at full speed (i.e. no runtime-dispatch within the loop). This is difficult with dynamic programs where parameters values, types or even algorithms themselves may be changed by the user during runtime. Moreover, pausing these loops, restarting them and making sure that no two loops are overlapping without the user meaning to, can also pose a challenge, and normally requires a lot of extra programming.

All loops in this package are managed by a Process struct. The manages functions that are looped over in a threaded loop, and handles pausing, restarting, type stability, loop iterations, etc. Loops may run indefinitely (i.e. untill stopped by the user), or for a fixed number of iterations. They hold arbitrary functions with arbitrary arguments. 

A function held by the loop is a description of what happens in one single iteration of the loop, typically where some arguments are mutated. They can then be accesses through the process itself (or just from an external reference if they are passed from outside),
The arguments are internally passed as a tuple for easier type stability, but this requires some care from the user when writing functions. Full details are given in [Processes](@ref).

We provide a factory function to create processes

```
process = makeprocess(func, repeats::Int = 0; prepare = preparefunc, kwargs...)
```

The argument `func` is any function like object that accepts a namedtuple holding all the arguments. It can be defined inline using the do syntax of Julia in the following way

```
process = makeprocess(repeats; prepare, a = Int[], b = 2, c = SomeStruct()) do args
    (;proc, a, b, c) =  args
    # Update the args
end
```

The arguments are passed as keyword arguments, so that they are accessible through named tuple unpacking within the function. The argument `proc` is included as a standard argument, mainly so that the function can be aware of the iteration number it's currently in. The iteration number can be accessed through

```
loopidx(proc)
```

## Windows

The interactive parts of the simulation are based on an abstraction on top of GLMakie in terms of windows. Windows manage their own updating, and their perhaps their own process. Windows are tracked inside a module, and when a window is closed, it will take care of the process running in the background to make sure everything is closed properly.

At the moment the only window facility offered is `lines_window(process)`, which expects a process that adds data to two vectors called `x` and `y`, which intuitively correspond to coordinates for the `x` and `y` coordinates for a datapoint. A process can be obtained from the function `linesprocess(func, number_of_repeats)`, which get an integer for the number of iterations a process goes through, where it will run indefinitely if `0` is given. The `linesprocess` just returns a normal process, but where the `x` and `y` coordinates are automatically produced in the `args` upon creation of the process.