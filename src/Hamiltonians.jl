module Hamiltonians
using ..InteractiveIsing
using ..InteractiveIsing: branchSim, setTuple


"""
Add factors to factor array defined in module scope
"""
macro addfactor(facnames...)
    for name in facnames
        push!(factors,eval(name))
    end
end

"""
Struct to define factors to be used in hamiltonian
expr: A string in julia syntax for the factor
symb: Define a unique symbol for the factor, if two factors use the same symbol
    a value can be used to pick any of the factors to be added over the others
val: Value for the symbol to pick one factor instead of the others
loop: Defines wether the factor is present in the loop or not. - more explanation needed -
"""
struct hFactor{T}
    expr::String
    symb::Symbol
    val::T
    loop::Bool
end

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
function setSimHType!(sim, pairs...; prt = false)
    sim.g.htype = editHType(sim.g.htype, pairs...)
    if prt
        println(sim.g.htype)
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

getHParam(g, symb) = getHParam(g.htype, symb)
export getHParam
export getHParamType


"""
Finds the expression that fits with the given symbol, value of the symbol,
and wether the factor is part of the loop or not
"""
function findExpr(symb, val, loop)
    for factor in factors
        if factor.symb == symb && factor.val == val && factor.loop == loop
            return factor.expr
        end
    end
    return ""
end

"""
Makes expression in form of string out of a set of symbols, values, 
and a predicate that tells the function wether the expression is part of
a loop over neighbors or not
"""
function buildExpr(loop, symbs, vals)
    str = string()

    for (idx, symb) in enumerate(symbs)
        term = findExpr(symb, vals[idx], loop)
        if term != ""
            if str != ""
                str *= " + "
            end
            str *= ("(@inbounds " * term * ")")
        end
        
    end
    return str
end

function getEFacExpr1(htype::Type{HType{Symbs,Vals}}) where {Symbs, Vals}
    exprvec = []

    line = "for conn in g.adj[idx] \n efactor +="
    line *= buildExpr(true, Symbs, Vals)

    line *= "end"

    push!(exprvec, Meta.parse(line))

    line = "return efactor"

    normalfactor = buildExpr(false, Symbs, Vals)

    line *= normalfactor != "" ? "+ "*normalfactor : ""

    push!(exprvec, Meta.parse(line))

    expr = Expr(
        :block,
        :(efactor = 0),
        exprvec...
    ) 
end
export getEFacExpr1

function getEFacExprNN(htype::Type{HType{Symbs,Vals}}) where {Symbs, Vals}
    exprvec = []

    line = "list = g.adj[idx]"

    expr = Expr(
        :block,
        Meta.parse(line),
        :(return -sum(connW.(list).* @inbounds (@view g.state[connIdx.(list)])))
        # :(return -sum(connW.(list).* @inbounds (g.state[connIdx.(list)])))
    ) 
end
export getEFacExprNN

"""
Get the energy factor (where we define E === σ_i Σ_j fac_j) for the state
the function is dispatched on the graph g, the idx i and the type of the Hamiltonian
which may be generated by the function generateHType(Symbs...).
"""
@generated function getEFactor(g, idx, htype::HType{Symbs,Vals}) where {Symbs, Vals}
    # if getHParamType(htype, :NN) == 1
    #     exp = getEFacExpr1(htype)
    # else
    #     exp = getEFacExprNN(htype)
    # end

    exp = getEFacExpr1(htype)

    return exp
end
export getEFactor


@generated function genE1(g, idx, htype::HType{Symbs,Vals}) where {Symbs, Vals}
    return getEFacExpr1(g, idx, htype)
end
export genE1

@generated function genENN(g, idx, htype::HType{Symbs,Vals}) where {Symbs, Vals}
    return getEFacExprNN(g, idx, htype)
end
export genENN


unweightedloop = hFactor("-g.state[connIdx(conn)]" , :Weighted, false, true)
weightedloop = hFactor("-connW(conn)*g.state[connIdx(conn)]", :Weighted, true, true)
magfac = hFactor("-g.d.mlist[idx]", :MagField, true, false)
clampfac = hFactor("g.d.clampfac[idx]*g.state[idx]", :Clamp, true, false)
defects = hFactor("", :Defects, false, false)
NN = hFactor("", :NN, 1, false)

factors = [];

@addfactor unweightedloop weightedloop magfac clampfac defects NN

end

# list = g.adj[idx]
# sum(connW.(list).* @inbounds (@view g.state[connIdx.(list)]))