struct IsingGraphCopy{G, T} <: AbstractIsingGraph{T}
    graph::G
    state::Vector{T}

    function IsingGraphCopy(graph::G; init = nothing)
        T = eltype(graph)
        state = similar(graph.state)
        if isnothing(init)
            copy!(state, graph.state)
        elseif init == :Random
            state .= initRandomState(graph)
        elseif init == :Zero
            state .= zero(T)
        else
            state .= init
        end
        
        new{typeof(graph), T}(graph, state) 
    end
end



