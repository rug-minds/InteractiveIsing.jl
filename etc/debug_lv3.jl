using SparseArrays, LoopVectorization

include("src/Utils/StaticFill.jl")

# Verify our overrides are installed
println("=== Verifying overrides ===")

# Check instruction! dispatch
sf = StaticFill(1.0, 100)
v = rand(100)

# Create a dummy LoopSet to test instruction!
# Actually, let's just check if the method exists
ms = methods(_LV.instruction!, Tuple{_LV.LoopSet, typeof(deferred_getindex)})
println("instruction! methods for typeof(deferred_getindex): ", length(ms))
for m in ms
    println("  ", m)
end

# Check instruction_cost dispatch
println("\n=== Testing instruction_cost ===")
cheap_instr = _LV.Instruction(:CHEAPCOMPUTE, :test)
opaque_instr = _LV.Instruction(Symbol(""), :test)
lv_instr = _LV.Instruction(:LoopVectorization, :getindex)

println("CHEAPCOMPUTE cost: ", _LV.instruction_cost(cheap_instr))
println("Opaque cost:       ", _LV.instruction_cost(opaque_instr))
println("LV getindex cost:  ", _LV.instruction_cost(lv_instr))

# Now let's trace what instruction! actually returns for deferred_getindex
# We need a real LoopSet for this
println("\n=== Looking at what @turbo actually generates ===")

# Let's intercept the instruction creation by temporarily wrapping
call_count = Ref(0)
original_instruction_cost = _LV.instruction_cost

function _LV.instruction_cost(instr::_LV.Instruction)
    result = if instr.mod === :LoopVectorization
        _LV.COST[instr.instr]
    elseif instr.mod === :CHEAPCOMPUTE
        _LV.InstructionCost(-3.0, 0.5, 3, 0)
    else
        _LV.OPAQUE_INSTRUCTION
    end
    # Log all calls with their mod
    if instr.instr !== :identity && instr.instr !== :getindex && instr.instr !== :setindex! &&
       !startswith(string(instr.instr), "v") && instr.instr !== :add_fast && instr.instr !== :mul_fast
        println("  instruction_cost(mod=$(instr.mod), instr=$(instr.instr)) => $(result)")
    end
    return result
end

sp_nzval = rand(10)
sp_rowval = rand(1:100, 10)

println("Compiling test_deferred_sf...")
function test_deferred_sf(sf, sp_nzval, sp_rowval, n)
    tot = 0.0
    @turbo for ptr in 1:n
        j = sp_rowval[ptr]
        wij = sp_nzval[ptr]
        tot += wij * deferred_getindex(sf, j)
    end
    return tot
end
println("Calling test_deferred_sf...")
result = test_deferred_sf(sf, sp_nzval, sp_rowval, 10)
println("Result: $result")

# Also check how many methods instruction! has for Function subtypes
println("\n=== All instruction! methods for Function ===")
ms_all = methods(_LV.instruction!)
for m in ms_all
    s = string(m)
    if occursin("Function", s) || occursin("deferred", s)
        println("  ", m)
    end
end
