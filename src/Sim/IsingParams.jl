mutable struct IsingParams
    updates::Int64

    shouldRun::Bool
    isRunning::Bool

    started::Bool

    circ::Vector{Tuple{Int16, Int16}}

    colorscheme::ColorScheme
end

IsingParams(;initbrushR = 1, colorscheme = ColorSchemes.viridis) = 
    IsingParams(
        0,
        true, 
        true, 
        false,
        # Circ
        getOrdCirc(Int32(initbrushR)),
        #ColorScheme
        colorscheme
    )