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
    type::Symbol
end

"""
Main function that generated the expression for the energy factor term
This factor is used for the generated function below
The reason this is a seperate function is for debugging
"""
function getEFacExpr(htype::Type{HType{Symbs,Vals}}) where {Symbs, Vals}
    exprvec = []

    line = "for conn in gadj[idx] \n efactor +="
    line *= buildExpr(:FacLoop, Symbs, Vals)

    line *= "end"

    push!(exprvec, Meta.parse(line))

    line = "return efactor"

    normalfactor = buildExpr(:FacTerm, Symbs, Vals)
    
    # Check if empty otherwise add a plus and the factor
    line *= normalfactor != "" ? "+ "*normalfactor : ""

    push!(exprvec, Meta.parse(line))

    expr = Expr(
        :block,
        :(efactor = 0),
        exprvec...
    ) 
end
export getEFacExpr


#= Main Energy Factor Function =#
"""
Get the energy factor (where we define E === σ_i Σ_j fac_j) for the state
the function is dispatched on the graph g, the idx i and the type of the Hamiltonian
which may be generated by the function generateHType(Symbs...).
"""
@generated function getEFactor(g, gstate, gadj, idx, htype::HType{Symbs,Vals})::Float32 where {Symbs, Vals}

    exp = getEFacExpr(htype)

    return exp
end
export getEFactor