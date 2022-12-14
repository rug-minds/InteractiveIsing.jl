export HType
struct HType{Symbs, Vals} end

function HType(pairs...)
    symbs = getUniqueParams(factors)
    vals = []

    for symb in symbs
        append!(vals, typeof(getFirstVal(symb))(0))
        for pair in pairs
            if pair[1] == symb
                vals[end] = pair[2]
            end
        end
    end

    return HType{symbs,tuple(vals...)}()
end
export HType

function getUniqueParams(factors)
    # Make a set of parameters for hamiltonian, ordered
    # by the order they are added by the macro addfactor
    paramset = tuple()
    for factor in factors
        symb = factor.symb
        for el in paramset
            if el == symb
                @goto cont
            end
        end
        paramset = (paramset..., symb)
        @label cont
    end
    return  paramset
end

function getFirstVal(symb)
    for factor in factors
        if factor.symb == symb
            return factor.val
        end
    end
    return false
end

"""
Generate a struct of the type: struct HType{Symbs, Vals} end
"""
function generateHType(vals...)
    symbs = getUniqueParams(factors)

    if length(vals) > length(symbs)
        error("Cannot be higher than number of symbols")
    end

    vals = (vals...,repeat([false], length((length(vals)+1):length(symbs)))...)

    return HType{symbs,vals}()

end
export generateHType

# not used
"""
Set hamiltonion params by providing a set of vals that are applied in order
"""
function editHType(htype::HType{Symbs, Vals}, vals...) where {Symbs, Vals}

    if length(vals) > length(Symbs)
        error("Cannot be higher than number of symbols")
    end

    vals = (vals..., Vals[(length(vals)+1):end]...)

    return HType{Symbs,vals}()
end

"""
Set hamiltonion params by providing a set of pairs of symbols and vals
"""
function editHType(htype::HType{Symbs, Vals}, pairs::Pair ...) where {Symbs, Vals}
    if length(pairs) > length(Symbs)
        error("Cannot be higher than number of symbols")
    end

    # Make vector from tuple
    vals = Vals

    # Edit symbols 
    for pair in pairs
        for (idx,symb) in enumerate(Symbs)
            if pair[1] == symb
                vals = setTuple(vals, idx, pair[2])
                @goto nextpair
            end
        end
        @label nextpair
    end
        
    return HType{Symbs,vals}()

end
export editHType

function editHType!(g, pairs...)
    g.htype = editHType(g.htype, pairs...)
end

export editHType!

export setSimHType!
function setSimHType!(sim, g, pairs...; prt = false)
    htype(g) = editHType(htype(g), pairs...)
    if prt
        println(htype(g))
    end
    branchSim(sim)
end

function paramIdx(htype::HType{Symbs,Vals}, symb) where {Symbs,Vals}
    idx = 1
    for symbol in Symbs
        if symb == symbol
            break
        end
        idx +=1
    end

    if idx > length(Symbs)
        error("Symbol not found")
        idx = length(Symbs)
    end

    return idx
end

"""
Get value of parameter of HType
"""
function getHParam(htype::HType{Symbs,Vals}, symb) where {Symbs, Vals}
    idx = paramIdx(htype, symb)

    return Vals[idx]
end

function getHParamType(htype::Type{HType{Symbs,Vals}}, symb) where {Symbs, Vals}
    return getHParam(HType{Symbs,Vals}(), symb)
end

getHParam(g, symb) = getHParam(htype(g), symb)
export getHParam
export getHParamType