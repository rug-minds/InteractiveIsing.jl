struct IsingGraphCopy{G, T} <: AbstractIsingGraph{T}
    graph::G
    state::Vector{T}

    function IsingGraphCopy(graph::G; init = nothing)
        T = eltype(graph)
        state = similar(InteractiveIsing.state(graph))
        if isnothing(init)
            copy!(state, InteractiveIsing.state(graph))
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


