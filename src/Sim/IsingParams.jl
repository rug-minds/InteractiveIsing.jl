mutable struct IsingParams
    updates::Int64

    shouldRun::Bool
    isRunning::Bool

    started::Bool

    shouldQuit::Bool

    circ::Vector{Tuple{Int16, Int16}}

    colorscheme::ColorScheme
end

IsingParams(;initbrushR = 1, colorscheme = ColorSchemes.viridis) = 
    IsingParams(
        # Updates
        0,
        true, 
        # shouldRun
        true, 
        # isRunning
        false,
        # shouldQuit
        false,
        # Circ
        getOrdCirc(Int32(initbrushR)),
        #ColorScheme
        colorscheme
    )