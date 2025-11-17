"""
This version gets rid of the args, but stores the location in the expression where the parameter ref was found
"""
struct NewParameterRef{Symb, indices, F, Location} <: AbstractParameterRef end

function getlocation(ex, symb)
    loc = nothing
    MacroTools.postwalk(x -> @capture(x, symb = location_) ? begin loc = location; x end : x, ex)
    return loc
end

macro NewParameterRefs(ex)
   
    MacroTools.postwalk(x -> @capture(x, p_[i__]) ? begin println("found param ", ParameterRef(p, getlocation(ex, p), i...), " in expression: ", ex); x end : x, ex)

    println("Ex: ", ex)
    @capture(ex, function fname_(a__) body_ end)

    println("Body: ", body)

    returnstatement = nothing
    MacroTools.postwalk(x -> @capture(x, return rt_) ? begin returnstatement = rt; x end : x, body)
    
    println("Returnstatement: ", returnstatement)
    return esc(:(
        function $fname($(a...))
            $returnstatement
        end))
end

function NewParameterRef(symb, location, indices ...; func = identity)
    NewParameterRef{symb, indices, typeof(func), location}()
end

export @NewParameterRefs

