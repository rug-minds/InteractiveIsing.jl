using LoopVectorization

lv_src = joinpath(pkgdir(LoopVectorization), "src")

# Read graphs.jl and find all functions that reference FUNCTIONSYMBOLS
println("=== graphs.jl: FUNCTIONSYMBOLS references ===")
lines = readlines(joinpath(lv_src, "graphs.jl"))
for (i, line) in enumerate(lines)
    if occursin("FUNCTIONSYMBOLS", line)
        # Show context: 5 lines before and 15 after
        for j in max(1, i-5):min(length(lines), i+15)
            marker = j == i ? ">>>" : "   "
            println("$marker $j: $(lines[j])")
        end
        println()
    end
end

# Also look at how operations/compute ops call the function
println("\n=== graphs.jl: 'instruction' near opaque function handling ===")
for (i, line) in enumerate(lines)
    if occursin("gensym", line) && (occursin("instruction", lowercase(line)) || occursin("Instruction", line) || occursin("preamble", line) || i > 1410 && i < 1430)
        for j in max(1, i-3):min(length(lines), i+3)
            marker = j == i ? ">>>" : "   "
            println("$marker $j: $(lines[j])")
        end
        println()
    end
end

# Check the add_compute function which handles compute ops
println("\n=== graphs.jl: add_compute / compute-op handling ===")
for (i, line) in enumerate(lines)
    if occursin("add_compute", line) && occursin("function", line)
        for j in i:min(length(lines), i+60)
            println("  $j: $(lines[j])")
        end
        println()
    end
end

# Also check how opaque functions are handled in the lowering phase  
println("\n=== lower*.jl: callexpr usage ===")
for fname in ["lower_compute.jl", "lowering.jl"]
    fpath = joinpath(lv_src, fname)
    if isfile(fpath)
        flines = readlines(fpath)
        for (i, line) in enumerate(flines)
            if occursin("callexpr", line)
                for j in max(1, i-2):min(length(flines), i+2)
                    marker = j == i ? ">>>" : "   "
                    println("$marker $fname:$j: $(flines[j])")
                end
                println()
            end
        end
    end
end
