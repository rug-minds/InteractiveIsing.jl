using SparseArrays, LoopVectorization

include("src/Utils/StaticFill.jl")

# Show what instruction! creates for deferred_getindex
# by inspecting the LoopSet
sf = StaticFill(1.0, 100)
v = rand(100)
sp_nzval = rand(10)
sp_rowval = rand(1:100, 10)

# Use LoopVectorization's internal API to see the LoopSet
println("=== FUNCTIONSYMBOLS lookup for deferred_getindex ===")
key = typeof(deferred_getindex)
if haskey(LoopVectorization.FUNCTIONSYMBOLS, key)
    instr = LoopVectorization.FUNCTIONSYMBOLS[key]
    println("  Found: mod=$(instr.mod), instr=$(instr.instr)")
else
    println("  NOT in FUNCTIONSYMBOLS (will use opaque path)")
end

# Let's also look at what callexpr produces for various mod values
println("\n=== callexpr behavior ===")
using LoopVectorization: Instruction
for mod in [Symbol(""), :LoopVectorization, :Main, :deferred_getindex]
    instr = Instruction(mod, :deferred_getindex)
    ce = LoopVectorization.callexpr(instr)
    println("  mod=$mod => $ce (type: $(typeof(ce)))")
end

# Now let's look at what happens when we actually compile a @turbo loop
println("\n=== Inspecting generated code ===")

# The real question: what does the generated let block look like?
# Let's use @turbo_debug or print the lowered code

# First let's check if the loop even works in working state
function test_deferred_sf(sf, sp_nzval, sp_rowval, n)
    tot = 0.0
    @turbo for ptr in 1:n
        j = sp_rowval[ptr]
        wij = sp_nzval[ptr]
        tot += wij * deferred_getindex(sf, j)
    end
    return tot
end

result = test_deferred_sf(sf, sp_nzval, sp_rowval, 10)
println("Working result: $result")

# Now let's look at what instruction LV actually creates
# by using @turbo_debug (if available) or @code_lowered
println("\n=== Looking at LV internals for an opaque function ===")

# Check the graphs.jl instruction! function signature
# When opaque: gensym → pushpreamble → Instruction(Symbol(""), gensym)
# Let's figure out what the gensym looks like

# Actually, let me check the real error. With FUNCTIONSYMBOLS[typeof(deferred_getindex)] set,
# instruction! should bypass the opaque path. But then callexpr generates 
# Expr(:call, :deferred_getindex) which should resolve in the caller's scope.
# But the error says "not defined in LoopVectorization".
# 
# Let me check: does LV eval the generated code in its own module?

println("\n=== Checking LV code generation module context ===")
# Look at condense_loopset.jl line 1179 where the error occurs
# That's likely where the generated code is being eval'd
lv_src = joinpath(pkgdir(LoopVectorization), "src")
println("LV source dir: $lv_src")

# Read the relevant part of condense_loopset.jl around line 1179
lines = readlines(joinpath(lv_src, "condense_loopset.jl"))
println("\n=== condense_loopset.jl around line 1179 ===")
for i in max(1, 1170):min(length(lines), 1195)
    println("  $i: $(lines[i])")
end

# Also check: where does setup_call or similar eval the generated expression?
println("\n=== Searching for eval or module context in condense_loopset.jl ===")
for (i, line) in enumerate(lines)
    if occursin("@eval", line) || occursin("Core.eval", line) || occursin("Base.eval", line) || (occursin("eval(", line) && !occursin("#", split(line, "eval(")[1]))
        println("  $i: $(strip(line))")
    end
end

# Check setup_call_noinline and related functions
println("\n=== Looking for the function that generates the turbo loop body ===")
for (i, line) in enumerate(lines)
    if occursin("setup_call", line) && (occursin("function", line) || occursin("@generated", line))
        println("  $i: $(strip(line))")
    end
end

# The key question: what module scope is the generated expression eval'd in?
# Let's check if there's a @generated function that uses the expressions
println("\n=== @generated functions in condense_loopset.jl ===")
for (i, line) in enumerate(lines)
    if occursin("@generated", line)
        # Show context
        for j in i:min(i+5, length(lines))
            println("  $j: $(lines[j])")
        end
        println()
    end
end
