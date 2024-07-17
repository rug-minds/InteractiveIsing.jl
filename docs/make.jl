push!(LOAD_PATH,"../src/")

using Documenter

makedocs(sitename="InteractiveIsing Documentation",
        pages = [ 
            # "Index" => "index.md",
            "General Usage" => "usage.md",
            "Details" => [
                "IsingGraphs" => "IsingGraphs.md",
                "Indexing" => "Indexing.md",
                "WeightGenerators" => "WeightGenerator.md",
                "Generating Adjacency Lists" => "GeneratingAdj.md",
                "Defects" => "Defects.md",
                "Loops" => "Loops.md",
                "Parameters" => "Parameters.md",
                "Algorithms" => "Algorithms.md",
                "Hamiltonians" => "Hamiltonians.md",
                "Analysis" => "Analysis.md",
                "Processes" => "Processes.md",
                "Topology" => "Topology.md",
            ]

        ]
        )

deploydocs(
    repo = "github.com/rug-minds/InteractiveIsing.jl.git",
)