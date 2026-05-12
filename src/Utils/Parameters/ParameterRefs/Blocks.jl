
### REFMULT
###
# These function transform a parameterref method call into a number
# I.e. fills and/or contractions of parameterrefs
###
function generate_block(rc::RefMult{Refs}, argstype, idxs = (;), precision = nothing, assignments = nothing) where Refs
    bmodel = get_blockmodel(rc, argstype, idxs)
    debugmode = argstype isa NamedTuple

    first = false
    if isnothing_generated(assignments) # Get the assignemnts of the base prefts
        assignments = assignment_map(rc, argstype, idxs)
        first = true
        debugmode && println("\t", "first assignment map")
    end

    filled_idxs = index_names(idxs)

    if debugmode
        println("generate_block refmult: $(rc)")
        println("\t", "bmodel: $bmodel")
    end
    if bmodel == BlockModel()
        block_sym = gensym("rm")
        blocks = [generate_block(p, argstype, idxs, precision, assignments) for p in get_prefs(rc)]
        blocknames = [Symbol(block_sym, :block_, i) for i in 1:length(blocks)]
        block_assignments = [Expr(:(=), blocknames[i], blocks[i]) for i in 1:length(blocks)]
        blockmult = Expr(:call, :*, blocknames...)
        
        return quote
            $(first ? unpack_keyword_expr(filled_idxs, :idxs) : :())
            $((first ? unique_assignments(assignments) : (:(),))...)
            $(block_assignments...)
            $blockmult
        end
    else
        if first
            return quote
                $(unpack_keyword_expr(filled_idxs, :idxs))
                $(unique_assignments(assignments)...)
                $(bmodel(rc, argstype, idxs, precision, assignments))
            end
        end
        return bmodel(rc, argstype, idxs, precision, assignments)
    end
end



function generate_block(rr::RefReduce, argstype, idxs = (;), precision = nothing, assignments = nothing)
    debugmode = argstype isa NamedTuple
    if debugmode
        println("generate_block refreduce: $(rr)")
    end

    first = false
    if isnothing_generated(assignments) # Get the assignemnts of the base prefts
        assignments = assignment_map(rr, argstype, idxs)
        first = true
        debugmode && println("\t", "first assignment map")
    end

    if !first
        assignments = submap(assignments, rr)
    end

    prefs = get_prefs(rr)
    # ind = ref_indices(rr)
    filled_indices = index_names(idxs)
    # contract_ind = idx_subtract(ind, filled_indices)

    # Name for the return
    totalname = gensym(:total)


    exps = nothing
    if ispure(rr) #Treat them all like a sum in turbo, small optimisation if possible
        if debugmode
            println("\t", "Pure mode refreduce")
        end
        block_sym = gensym("refreduce")

        flat_prefs = flatprefs(rr)
        loop_inds = loop_idxs(rr, idxs)
        assignmentmap = submap(assignments, rr)
        names = get_names(assignmentmap)
        u_assignments = unique_assignments(assignmentmap)
        
        red_exp = wrapF(rr, operator_reduce_exp_single(names, get_reduce_fs(rr)))
        total_add = Expr(:+=, totalname, red_exp)


        # TODO: Capture this in a block model
        # vec_accesses = struct_ref_exp(rr) ## Filter out veclikes and scalar likes later
        # ref_symbs = ref_symb.(prefs)
        # vec_names = [Symbol(block_sym, :vec_, ref_symbs[i]) for i in 1:length(prefs)]
        # vec_inds = [ref_indices(prefs[i]) for i in 1:length(prefs)]
        # assignments = [Expr(:(=), vec_names[i], vec_accesses[i]) for i in 1:length(prefs)]

        # Get the reduce fs
        # reduce_fs = get_reduce_fs(rr)
        # # nplusses = num_plusses(rr)
        # # nminuses = num_minuses(rr)
        # prefs = get_prefs(rr)
        # wrapped_vecrefs = [wrapF(rr[1], Expr(:ref, vec_names[1], vec_inds[1]...)), 
        # (reduce_fs[i] == (+) ? wrapF(prefs[i+1], Expr(:ref, vec_names[i+1], vec_inds[i+1]...)) : 
        # Expr(:call, :-, wrapF(prefs[i+1], Expr(:ref, vec_names[i+1], vec_inds[i+1]...))) for i in 1:length(reduce_fs))...]
        # F_call = wrapF(rr, Expr(:call, :+, wrapped_vecrefs...))
        # total_add = Expr(:+=, totalname, F_call)

        # push!(reduce_vecrefs.args, Expr(:+=, :total, Expr(:call, :+, (partialf_exp(getF(rr[i]), Expr(:ref, Symbol(:vec, i), :i)) for i in 1:nplusses)...)))
        # (nminuses != 0) && push!(reduce_vecrefs.args, Expr(:-=, :total, Expr(:call, :+, (partialf_exp(getF(rr[i]), Expr(:ref, Symbol(:vec, i), :i)) for i in nplusses+1:length(prefs))...)))
        
        exps = quote
            $(nested_turbo_wrap(total_add, (:(axes($(vec_names[1]), $i_ind)) for i_ind in 1:length(loop_inds)) |> collect, loop_inds))
            # TODO FIX GETTING THE RIGHT AXES WHEN THERE ARE VECREFS WITH DIFFERENT DIMENSIONS
        end
    else #Fallback is just reduce the separate blocks
        if debugmode
            println("\t", "Fallback mode refreduce")
        end
        block_sym = gensym("rr")
        block_exps = generate_block.(prefs, Ref(argstype), Ref(idxs), Ref(precision), Ref(assignments))
        blocknames = [Symbol(block_sym, :block_, i) for i in 1:length(prefs)]
        reduce_exp = :($totalname += $(wrapF(rr, operator_reduce_exp_single(blocknames, get_reduce_fs(rr)))))
        if debugmode
            println("\tget_reduce_fs(rr): $(get_reduce_fs(rr))")
        end
        exps = quote
            $([Expr(:(=), blocknames[i], block_exps[i]) for i in 1:length(prefs)]...)
            $(reduce_exp)
        end        
    end

    # exp = quote 
    #     $(unpack_keyword_expr(filled_indices, :idxs))
    #     $totalname = $(getzero_exp(rr, precision))
    #     $(exps)
    #     $totalname
    # end

    exp = quote
        $(first ? unpack_keyword_expr(filled_indices, :idxs) : :())
        # $totalname = $(getzero_exp(rr, precision))
        $((first ? unique_assignments(assignments) : (:(),))...)

        $totalname = $(getzero_exp(precision, get_names(assignments)...))
        $(exps)
        $totalname
    end
    return exp
end


