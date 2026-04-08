using ChainRulesCore

function ChainRulesCore.rrule(layer::LayeredIsingGraphLayer, x, ps, st)
    # ---------- primal free phase ----------
    g = GraphFromInit(st.graph, ps)



    off!(g.index_set, layer.input_layer)
    state(g[1]) .= x



    proc = st.relax_routine
    minus_capture = st.minus_capture
    plus_capture = st.plus_capture

    # reset!(proc, Input(Metropolis(), state = g))
    g.hamiltonian[4].β = layer.β #TODO: Find a way to refer to \beta cleanly
    c = run(proc, Input(Metropolis(), state = g))
    plus_state = c[plus_capture].captured
    minus_state = c[minus_capture].captured
    
    output_idxs = layer.output_layer
    yplus = @view plus_state[output_idxs]
    yminus = @view minus_state[output_idxs]

    project_ps = ProjectTo(ps)

    function pullback(ΔΩ)
        ΔΩ = unthunk(ΔΩ)
        ȳ, _ = ΔΩ

        βf = layer.β

        s_plus  = nudged_from_free_state(layer, s_free, x, ps, st, ȳ, +βf)
        s_minus = nudged_from_free_state(layer, s_free, x, ps, st, ȳ, -βf)

        ∂ps = @thunk begin
            dweights = similar(ps.weights)
            for (k, (i, j)) in enumerate(each_trainable_edge(st.graph))
                dweights[k] =
                    (Float32(s_plus[i]) * Float32(s_plus[j]) -
                     Float32(s_minus[i]) * Float32(s_minus[j])) / (2f0 * βf)
            end

            dbiases = similar(ps.biases)
            @inbounds for i in eachindex(ps.biases)
                dbiases[i] = (Float32(s_plus[i]) - Float32(s_minus[i])) / (2f0 * βf)
            end

            dα = similar(ps.α_i)
            @inbounds for i in eachindex(ps.α_i)
                dα[i] = (Float32(s_plus[i])^2 - Float32(s_minus[i])^2) / (2f0 * βf)
            end

            project_ps((weights = dweights, biases = dbiases, α_i = dα))
        end

        ∂layer = NoTangent()
        ∂x     = ZeroTangent()
        ∂st    = NoTangent()

        return ∂layer, ∂x, ∂ps, ∂st
    end

    return (y, st_out), pullback
end