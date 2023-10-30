"""
Used for decouple!
"""
function findcallback(input, obs)
    for (l_idx, listener_pair) in enumerate(input.listeners)
        listener = listener_pair[2]
        if typeof(listener) == Observables.MapCallback
            if listener.result == obs
                return l_idx
            end
        end
    end
    return nothing
end

"""
Map observables to new ones that are coupled_obs
This means the input observables are stored in obs.inputs
Inputs can be decoupled from the output observable by calling decouple! on the output
"""
function liftcouple(f, obs::AbstractObservable, args...)
    newob = lift(f, obs, args...)
    append!(newob.inputs, [obs, args...])
    return newob
end
"""
Decouple an output observable made with liftcouple from its inputs
"""
function decouple!(obs::AbstractObservable)
    for (i_idx, input_ob) in enumerate(obs.inputs)
        if typeof(input_ob) <: Observable
            callback_idx = findcallback(input_ob, obs)
            deleteat!(input_ob.listeners, callback_idx)
            deleteat!(obs.inputs, i_idx)
            return true
        end
    end
    return false
end
