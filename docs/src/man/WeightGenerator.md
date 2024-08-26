# WeightGenerator

Weight generators are structs that use some metaprogramming to turn a description of a function dependent on coordinates in the lattice of an IsingLayer into a weight.
The details of weightgenerators may change, but the interface should remain fairly stable.

A `WeightGenerator` holds descriptions of the following things:

* A julia anonymous function that describes how to generate the weight from the following arguments:
    `[dr,dx,dy,dz,x,y,z]` where `dr` is the relative distance between two spins, `dx, dy, dz` are displacements in the `x, y` and `z` coordinates respectively
    `x, y, z` are the averaged coordinate of the two spins, starting at one at the top left of a lattice, where `y` counts up going downwards.

    The function given may accept any combination of these coordinates

* A julia anonymous function describing how to generate the self energy of a spin (i.e. $J_ii$) from the following arguments:
    `[x, y, z]` which are the lattice coordinates of a spin.

* A description of a distribution from which a number is sampled which is **added to** the generated weight.
    I.e. `Normal(μ,σ)` would add a random number generated from a normal distribution with mean `μ` and standard deviation `σ` to the weight generated from the function above.
 
* A description of a distribution from which a number is sampled which is **multiplied with** the generated weight.
    I.e. using `[-1,1]`, a random sign can be will be sampled with equal probability which is mulitplied with the weight.

* The amount of nearest neighbors for which a weight needs to be generated. This may be given as a single integer or a tuple of integers. A tuple must be given to have at least the amount of dimensions of the layer the generator will be applied to. The numbers in the tuple will represent the amount of nearest neighbors that will be taken into account when generating connections, let's call the entries $(N_x, N_y, N_z)$, where the numbers correspond to the $x,y$ and $z$ axis respectively. This means that in the $x$ direction, only spins with a relative $x$ distance of maximally $N_x$ will be considered when generating weights, and so on for $y$ and $z$.

!!! note
    At first sight this might seem redundant, because we can program a distance cutoff in the generating function that we supply, i.e. `(dr) -> dr < 3 ? somefunc(dr) : 0` which ensures that all weights for distances greater than 3 are zero. However, julia code cannot know beforehand wether some arbitrary function causes a distance cutoff. Thus, all combinations of spins would need to be checked to see if they produce a zero weight. This is computationally quite prohibitive because every combination of two spins has to be checked. Thus the normal computation time for generating the weights would be on the order of $N_{spins}^2$, where $N_{spins} \sim L^d$, where L is the dimension of the material. With this method we have scaling of $N_{spins} * {NN}^d$, where typically ${NN}^d << L^d$, which drastically improves computation time for most type of local connectivity, where the connections are negligible after a limited distance anyway.

## Why Metaprogramming?

Metaprogramming allows for flexible definition of the weights, like choosing which arguments are neccesary and chooshing whether or not to have a self energy or distirbution term, while still being performant. A user can now just write `(dx, dy) -> somefunc(dx,dy)` which will be handled by the metaprogramming code. If metaprogramming wouldn't be used, any description of the function would need to include all possible arguments, even if they are not used `(dr,dx,dy,dx,x,y,z) -> somefunc(dx,dy)` to ensure consistent working with the functions in the module that generate the adjacency lists from these weights.

Performance is also key because the amount of weights generated for even a simple lattice might otherwise already be prohibitive in terms of computation time.

## Constructing WeightGenerators

Because we need metaprogramming capabities, weightgenerators are not generated through a normal julia function, but a macro, `@WG`, which returns a regular julia struct. It accepts the following

```
wg = @WG "(args) -> somefunc(...)" selfWeight = "(selfargs) -> someselffunc(...)" addDist = "Dist(...)" multDist = "Dist(...)" NN = 3
```

Where args is any combination of the arguments `[dr, dx, dy, dz, x, y, z]`, and selfargs is any combination of `[x,y,z]`. For the interpretations of the arguments see the first section [WeightGenerator](@ref).

The first argument and the keyword argument are required. Except for the argument `NN`, which is given as an int, all other arguments are given as strings, otherwise containing normal julia code.

The distributions may be given in two ways. The first is to provide any julia code that may be filled into `rand`. I.e. `Normal(μ,σ)` may be used as `rand(Normal(μ,σ))` to generate numbers from a normal distribution, or `[-1,1]` may be used as `rand([-1,1])` to generate an integer from that list with equal probability. We may also give any `rand(...)` function directly, for example to produce numbers from a list with inhomogeneous probability.

### Redundancy

There is a bit of redundancy built in for convenience's sake. The `addDist` and `multDist` arguments provide global distributions which also could have been given through the weight generating function itself. This however keeps the code a bit cleaner and easier to read.

## Getting the weight

We can get the weight from a weightgenerator with

```
getWeight(wg::WeightGenerator; dr = 0, dx = 0, dy = 0, dz = 0, x = 0, y = 0, z = 0)
```

where the arguments not used by the function given during construction simply have no effect.

We can get the self weight with

```
getSelfWeight(wg::WeightGenerator; x = 0, y = 0, z = 0)
```
