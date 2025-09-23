
# """
# Struct to generate weights for ising graphidxs
# """
# struct WeightGenerator{Func <: Function , SelfFunc <: Union{Nothing, Function}, AddFunc <: Union{Nothing, Function}, MultFunc <: Union{Nothing, Function}}
#     NN::Union{Int32, NTuple{2,Int32}, NTuple{3,Int32}}
#     func::Func
#     selfWeight::SelfFunc
#     addDist::AddFunc
#     multDist::MultFunc
#     funcstr::Union{Nothing,String}
#     selfstr::Union{Nothing,String}
#     addstr::Union{Nothing,String}
#     multstr::Union{Nothing,String}

#     # function 

#     """
#     Create a WeightGenerator
#     """
#     function WeightGenerator(NN, func, selfWeight = nothing, addDist = nothing, multDist = nothing, funcstr = "", selfstr = "", addstr = "", multstr = "")
#         new{typeof(func), typeof(selfWeight), typeof(addDist), typeof(multDist)}(Int32.(NN), func, selfWeight, addDist, multDist, funcstr, selfstr, addstr, multstr)
#     end
    

#     """
#     Copy and change
#     """
#     function WeightGenerator(wg::WeightGenerator; NN = nothing, func = nothing, selfWeight = nothing, addDist = nothing, multDist = nothing, funcstr = nothing, selfstr = nothing, addstr = nothing, multstr = nothing)
#         isnothing(NN) && (NN = wg.NN)
#         isnothing(func) && (func = wg.func)
#         isnothing(selfWeight) && (selfWeight = wg.selfWeight)
#         isnothing(addDist) && (addDist = wg.addDist)
#         isnothing(multDist) && (multDist = wg.multDist)
#         isnothing(funcstr) && (funcstr = wg.funcstr)
#         isnothing(selfstr) && (selfstr = wg.selfstr)
#         isnothing(addstr) && (addstr = wg.addstr)
#         isnothing(multstr) && (multstr = wg.multstr)

#         new{typeof(func), typeof(selfWeight), typeof(addDist), typeof(multDist)}(Int32.(NN), func, selfWeight, addDist, multDist, funcstr, selfstr, addstr, multstr)
#     end
# end

# @setterGetter WeightGenerator
# export WeightGenerator

# function Base.show(io::IO, wg::WeightGenerator)
#     println(io, "WeightGenerator with")
#     println(io, "\t NN: \t\t\t\t", wg.NN)
#     print(io, "\t func: \t\t\t\t", wg.funcstr)
#     if wg.selfstr != nothing
#         print(io, "\n\t Self Weight Function: \t\t", wg.selfstr)
#     end
#     if wg.addstr != nothing
#         print(io, "\n\t Additive Distribution: \t", wg.addstr)
#     end
#     if wg.multstr != nothing
#         print(io, "\n\t Multiplicative Distribution: \t", wg.multstr)
#     end
# end


# abstract type SelfType end
# struct Self <: SelfType end
# struct NoSelf <: SelfType end

# function SelfType(wg::WeightGenerator{A,SelfFunc,B,C}) where {A,SelfFunc,B,C}
#     if isa(SelfFunc, Type{Nothing})
#         return NoSelf()
#     else
#         return Self()
#     end
# end
# export SelfType



# # Can this be turned into a function, sinze we're using string?
# # Don't think so due to world age.
# """
# Create a WeightGenerator
# Accepted keywords are: NN, selfWeight, addDist, multDist, weightFunc
# Either: Either give a string with a function or
# give a previous weightfunc and supply keyword args to modify

# Done with a macro so that the function code can be recalled
# And the function itself doesn't have to be runtime generated
#     TODO:Switch to runtime generated functions?
# """
# const allowedargs_func = [:dr, :x, :y, :z, :dx, :dy, :dz]
# const allowedargs_self = [:x, :y, :dz]
# function args_to_str(args)
#     return join(string.(args), ",")
# end

# macro WeightGenerator(wg_or_func, kwargs...)
#     # Default Params
#     NN = 1

#     # allowedargs_func = [:dr, :x, :y, :dx, :dy]
#     # allowedargs_self = [:x, :y]

#     is_a_wg = false
#     # When a string is given
#     if isa(wg_or_func, String)
#         # Set defaults
#         selfWeight = addDist = multDist = nothing
#         selfstr = addstr = multstr = nothing
        
#         # Set kwargs
#         kwargs = (kwargs...,:(weightFunc = $wg_or_func))
#     else
#         is_a_wg = true
#         NN = func = selfWeight = addDist = multDist = funcstr = selfstr = addstr = multstr = nothing
#     end
#     # Get keyword arguments
#     if !isempty(kwargs)
#         params = prunekwargs(kwargs...)
#         func_key = selfWeight_key = addDist_key = multDist_key = nothing
#         # If keyword arguments get them
#         for exp in params
#             args = exp.args
#             symb = args[1]
#             val = args[2]
#             if symb == :NN
#                 NN = val
#             elseif symb == :selfWeight
#                 selfWeight_key = val
#             elseif symb == :addDist
#                 addDist_key = val
#             elseif symb == :multDist
#                 multDist_key = val
#             elseif symb == :weightFunc
#                 func_key = val
#             else
#                 error("Unknown keyword argument $symb")
#             end
#         end
#     else
#         func_key = selfWeight_key = addDist_key = multDist_key = nothing
#     end
    
#     # FUNC
#     if !isnothing(func_key)
#         funcstr = func_key
#         funcexpr = Meta.parse(func_key)
#         # get func argnames
#         argnames_func = method_argnames(last(methods(eval(funcexpr))))[2:end]
#         # Check if argnames only contain a subset of the symbols allowedargs_func
#         if !(all([arg ∈ allowedargs_func for arg in argnames_func]))
#             error("Function must only contain arguments $allowedargs_func")
#         end

#         # Get function body
#         funcbody = funcexpr.args[2]
#         func = quote @inline (;dr,x,y,z,dx,dy,dz) -> Float32($funcbody) end
#     end
    
#     # SELF WEIGHT
#     if !isnothing(selfWeight_key)
#         selfstr = selfWeight_key
#         selfWeightExpr = Meta.parse(selfWeight_key)
#         # println(selfWeightExpr)
#         # println(methods(eval(selfWeightExpr)))
#         # println(method_argnames(methods(selfWeightExpr)[]))

#         # get selfweight argnames
#         argnames_self = method_argnames(last(methods(eval(selfWeightExpr))))[2:end]
#         # Check if argnames only contain a subset of the symbols :i, :j
#         if !(all([arg ∈ allowedargs_self for arg in argnames_self]))
#             error("Self weight Function must only contain arguments $allowedargs_self")
#         end
        
#         # Get function body
#         selfweightbody = selfWeightExpr.args[2]

#         selfWeight = quote @inline (;x,y) -> Float32($selfweightbody) end
#     end

#     # ADDITIVE DISTRIBTION
#     if !isnothing(addDist_key)
#         addstr = addDist_key
#         if contains(addDist_key, "rand") || contains(addDist_key, "sample")
#             addDist = Meta.parse("@inline () -> Float32($addDist_key)")
#         else
#             addDist = Meta.parse("@inline () -> Float32(rand($addDist_key))")
#         end
#     end

#     # MULTIPLICATIVE DISTRIBTION
#     if !isnothing(multDist_key) 
#         multstr = multDist_key
#         if contains(multDist_key, "rand") || contains(multDist_key, "sample")
#             multDist = Meta.parse("@inline () -> Float32($multDist_key)")
#         else
#             multDist = Meta.parse( "@inline () -> Float32(rand($multDist_key))" )
#         end
#     end

#     if !is_a_wg
#         return esc(quote
#             WeightGenerator($NN, $func, $selfWeight, $addDist, $multDist, $funcstr, $selfstr, $addstr, $multstr)
#         end)
#     else
#         return esc(quote
#             WeightGenerator($wg_or_func, NN = $NN, func = $func, selfWeight = $selfWeight, addDist = $addDist, multDist = $multDist, funcstr = $funcstr, selfstr = $selfstr, addstr = $addstr, multstr = $multstr)
#         end)
#     end
# end
# macro WeightGenerator!(wg, args...) 
#     return esc(quote
#         $wg = @WeightGenerator $wg $(args...)
#     end)
# end
# var"@WG" = var"@WeightGenerator"
# var"@WG!" = var"@WeightGenerator!"
# export @WeightGenerator, @WeightGenerator!, @WG, @WG!

# @generated function getWeight(wg::WeightGenerator{Func, SelfFunc, AddFunc, MultFunc}; dr = 0, dx = 0, dy = 0, dz = 0, x = 0, y = 0, z = 0) where {Func, SelfFunc, AddFunc, MultFunc}
#     return Meta.parse("wg.func(;$(args_to_str(allowedargs_func)))"*(!isa(MultFunc, Type{Nothing})*"*wg.multDist()" * (!isa(AddFunc, Type{Nothing})*" + wg.addDist()")))
# end

# @generated function getSelfWeight(wg::WeightGenerator{Func, SelfFunc, AddFunc, MultFunc}; x, y, z) where {Func, SelfFunc, AddFunc, MultFunc}
#     return Meta.parse("wg.selfWeight(;$(args_to_str(allowedargs_self)))"*(!isa(MultFunc, Type{Nothing})*"*wg.multDist()" * (!isa(AddFunc, Type{Nothing})*" + wg.addDist()")))
# end
# export getWeight, getSelfWeight


# Base.zero(::Type{NTuple{N,T}}) where {N,T} = NTuple{N,T}(Base.zero(T) for i in 1:N)
# Base.zero(::Type{Tuple{T1,T2}}) where {T1,T2} = (Base.zero(T1), Base.zero(T2))


# const wg_isingdefault = @WeightGenerator "(dr) -> dr == 1" NN = 1
