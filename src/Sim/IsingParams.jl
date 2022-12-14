mutable struct IsingParams
    updates::Int64

    shouldRun::Bool
    isRunning::Bool

    started::Bool

    circ::Vector{Tuple{Int16, Int16}}
end

IsingParams(initbrushR) = 
    IsingParams(
        0,
        true, 
        true, 
        false,
        # Circ
        getOrdCirc(Int32(initbrushR))
    )