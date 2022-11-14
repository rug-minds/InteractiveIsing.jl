include("Hamiltonians.jl")

using InteractiveIsing
using Distributions

const sim = IsingSim(
    continuous = true, 
    graphSize = 500, 
    weighted = true;
    )

g = sim(false);

hType = generateHType(true,false,false)