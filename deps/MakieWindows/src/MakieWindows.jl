module MakieWindows
    include("../../StatefulAlgorithms/src/StatefulAlgorithms.jl")

    const windows = Dict{UUID,AbstractWindow}()

    using GLMakie
    using .StatefulAlgorithms

    include("Windows.jl")
end # module MakieWindows
