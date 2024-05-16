# module MCAlgorithms  
    using LoopVectorization, SparseArrays, MacroTools, RuntimeGeneratedFunctions
    export build_H

    abstract type MCAlgorithm <: Function end
    abstract type Hamiltonian end
    abstract type DerivedHamiltonian end

    const defined_derived = Dict{Type, Type}()

    include("ParamVal.jl")


    """
        Fallback required derived hamiltonian for an algorithm, 
        indicating that that the algorithm did not define a required derived hamiltonian.
    """
    function requires(::Type{<:MCAlgorithm})
        throw(ArgumentError("This algorithm did not define a required derived Hamiltonian."))
    end

    function prepare(::Type{Algo}, g; kwargs...) where {Algo <: MCAlgorithm}
        args_algo = (;_prepare(Algo, g; kwargs...)...)

        return args_algo
    end


    # A switch to avoid runtime dispatch on the layertype
    # Groups layeridxs based on type and then creates a fixed switch statement
    # Only dispatches on the type of tyhe layer.
    # Add dispatch on the layertrait if necessary
    @generated function layerswitch(@specialize(func), i, layers::LayerTuple, @specialize(args)) where LayerTuple
        code = Expr(:block)
        grouped_idxs, layer_idxs = group_idxs(layers)
  
        for group_idx in eachindex(grouped_idxs)
            upperbound_idx = last(grouped_idxs[group_idx])

            layertype = layers.parameters[layer_idxs[group_idx]]

            codeline = :(if i <= $upperbound_idx; return func(i, args, $layertype); end)

            push!(code.args, codeline)
        end
        push!(code.args, :(throw(BoundsError(structs, i))))
        return code
    end
    

    include("Utils.jl")
    include("Algorithms/Algorithms.jl")
    include("Hamiltonians/Hamiltonians.jl")
# end