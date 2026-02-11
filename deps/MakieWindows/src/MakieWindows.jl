module MakieWindows
    include("../../Processes/src/Processes.jl")

    const windows = Dict{UUID,AbstractWindow}()

    using GLMakie
    using .Processes

    include("Windows.jl")
end # module MakieWindows
