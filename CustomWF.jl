# Custom WeightFuncs

""" 
    Use this file to define custom weightfuncs.

    Preferred constructor to be used for WeightFunc:

        WeightFunc(func::Function, ; NN::Integer)

    Caveat: func should be defined as follows

    Function must depend only on a combination of arguments: dr, i and j
    where dr is the relative distance between to nodes, and i and j are the lattice coordinates

    It should be defined as follow: Either first define a regular func with any of the keyword arguments and end with _...
        
        function weightF(;dr, i , _...)

    Or define an anonymous function within the constructor
        
        (;i,j , _...) -> 

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
RadialWeightF = WeightFunc(
    (;dr, i,j,_...) -> radialFunc(pow ;dr, i,j),
    NN = 2
)