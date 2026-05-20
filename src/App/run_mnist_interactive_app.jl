using Pkg

Pkg.activate(joinpath(@__DIR__, "..", "..", "ext", "IsingLearning"))

include(joinpath(@__DIR__, "MNISTInteractiveApp.jl"))

if "--check" in ARGS
    println(MNISTInteractiveApp.smoke_check())
    exit(0)
else
    exit(MNISTInteractiveApp.julia_main())
end
