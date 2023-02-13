mutable struct IsingParams
    started::Bool

    updates::Int64

    circ::Vector{Tuple{Int16, Int16}}

    colorscheme::ColorScheme
end

IsingParams(;initbrushR = 1, colorscheme = ColorSchemes.viridis) = 
    IsingParams(
        # started
        false,
        # Updates
        0,
        # Circ
        getOrdCirc(Int32(initbrushR)),
        #ColorScheme
        colorscheme
    )