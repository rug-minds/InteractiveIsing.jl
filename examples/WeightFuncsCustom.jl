# Custom WeightFuncs

""" 
    Use this file to define custom weightfuncs.

    Preferred constructor to be used for WeightFunc:

        WeightFunc(func::Function, ; NN::Integer)

    Caveat: func should be defined as follows

    Function must depend only on a combination of arguments: dr, i and j
    where dr is the relative distance between to nodes, and i and j are the lattice coordinates

   Then either first define a regular func with any of the keyword arguments and end with _...
   Example: 
   
        function weightF(;dr, i , _...)
            ...
        end


    Or define an anonymous function within the constructor
    Example: 
        
        (;i,j , _...) -> ...

        
    Then use this in the constructor
    Example:

        # Default ising Function
        DefaultIsing() =  WeightFunc(
            (;dr, _...) -> dr == 1 ? 1. : 0., 
            NN = 1
        )

"""

# Radial weightfunc
function radialFunc(pow; dr, i, j, _...)
    return (1/dr^2)*1/((i-256)^2+(j-256)^2)^pow
end
pow = .25
radialWF = WeightFunc(
    (;dr, i,j,_...) -> radialFunc(pow ;dr, i,j),
    NN = 1
)

isingNN2 = WeightFunc(
    (;dr, i,j,_...) -> 1/dr,
    NN = 2
)

isingNN22 = WeightFunc(
    (;dr, i,j,_...) -> 1/(dr^2),
    NN = 2
)