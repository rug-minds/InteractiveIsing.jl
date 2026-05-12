
struct HashOrder <: Base.Ordering end

function Base.lt(::HashOrder, a, b)
    return hash(a) < hash(b)
end

export BlockModel
struct BlockModel{T} 
    function BlockModel(questions...)
        if isempty(questions)
            return new{Nothing}()
        end
        new{(questionsort(questions))}()
    end
    BlockModel() = new{Nothing}()
    BlockModel{Nothing}() = new{Nothing}()
    BlockModel(::Nothing) = BlockModel()

end #Just a val like wrapper

const NoBlockModel = BlockModel{Nothing}

const registered_questions = SortedSet{Function}(HashOrder())
const block_models = Dict{Tuple, BlockModel}()

questionsort(v) = sort(v, lt = (x,y) -> hash(x) < hash(y))

function register_blockmodel(questions...)
    questions = questionsort(questions)
    push!(registered_questions, questions...)
    block_models[questions] = BlockModel(questions...)
end

function get_blockmodel(apr::AbstractParameterRef, argstype, idxs)
    debugmode = argstype isa NamedTuple
    if debugmode
        println("get_blockmodel: $(apr)")
        println("\t", "idxs: $idxs")
    end
    evals = [false for _ in registered_questions]
    for (qidx, q) in enumerate(registered_questions)
        if debugmode
            println("\t", "evaluating $q")
        end
        if q(apr, argstype, idxs)
            debugmode && println("\t", "evaluated $q as true")
            evals[qidx] = true
        end
    end
    return get(block_models, tuple(collect(registered_questions)[evals]...), BlockModel())
end

#Fallback BlockModel
function (::BlockModel{Nothing})(pr, argstype, idxs, precision = nothing, assignments = nothing)
    generate_block(pr, argstype, idxs, precision, assignments)
end

###
# Traits
###
export L_vec_R_spmatrix, allPure
function L_vec_R_spmatrix(pr, argstype, filled_indices)
    retval = true

    # Left is vector, right is sparse matrix
        retval = retval && length(get_prefs(pr)) == 2
        idxs1 = ref_indices(pr[1])
        retval = retval && length(idxs1) == 1
        idxs2 = ref_indices(pr[2])
        retval = retval && length(idxs2) == 2
        retval = retval && idxs1[1] == idxs2[1]
        retval = retval && dereftype(pr[2], argstype) <: SparseMatrixCSC
    return retval

end

register_blockmodel(L_vec_R_spmatrix)

function allPure(pr, argstype, filled_indices)
    return ispure(pr)
end

register_blockmodel(allPure)

# function allsameindex(pr, argstype, filled_indices)
#     prefs = get_prefs(pr)
#     loop_inds = loop_idxs.(prefs, Ref(filled_indices))
#     retval = true
#     for inds_idx in 1:length(loop_inds)-1
#         if loop_inds[inds_idx] != loop_inds[inds_idx+1]
#             return false
#         end
#     end
#     return retval
# end

# register_blockmodel(allsameindex)


## Blockmodels

#### REFREDUCES
function (::typeof(BlockModel()))(rr::RefReduce, argstype, idxs, precision = nothing, assignments = nothing)
    prefs = get_prefs(rr)
    ind = ref_indices(rr)
    filled_indices = index_names(idxs)
    contract_ind = idx_subtract(ind, filled_indices)

    # Name for the return
    totalname = gensym(:total)

    block_exps = generate_block.(prefs, Ref(argstype), Ref(idxs), Ref(precision), Ref(assignments))
    blocknames = [Symbol("block_", i) for i in 1:length(prefs)]
    reduce_exp = :($totalname += $(operator_reduce_exp(blocknames, get_reduce_fs(rr))))
    exps = quote
        $([Expr(:(=), blocknames[i], block_exps[i]) for i in 1:length(prefs)]...)
        $(reduce_exp)
    end    
end


#### REFMULTS
function (::typeof(BlockModel(L_vec_R_spmatrix)))(rc::RefMult, argstype, filled_idxs, precision = nothing, assignments = nothing)
    loop_ind = loop_idxs(rc, filled_idxs)[1]
    mat_inds = ref_indices(rc[2])

    tloop = quote @turbo for ptr in nzrange(sp_matrix, $(mat_inds[2]))
        $(loop_ind) = sp_matrix.rowval[ptr]
        wij = sp_matrix.nzval[ptr]
        cumsum += wij * vector[$(loop_ind)]
        end
    end

    filled_idxs = index_names(filled_idxs)
    if !(:j ∈ filled_idxs)
        tloop = quote 
            for j in axes(sp_matrix, 2)
                $(tloop)
            end
        end
    end

    expr = quote
        $(unpack_keyword_expr(filled_idxs, :idxs))
        vector = $(struct_ref_exp(rc[1])...)
        sp_matrix = $(struct_ref_exp(rc[2])...)
        cumsum = zero((@inline promote_eltype(vector, sp_matrix)))
        # cumsum = zero(Float32)
        $tloop
        
        cumsum
    end
end


# """
# Fallback blockmodel for a refmult
# """
# function (::typeof(BlockModel(allPure)))(rc::RefMult, argstype, filled_idxs)
#     prefs = get_prefs(rc)
#     inds = ref_indices.(prefs)
#     symbs = ref_symb.(prefs)

#     names = [gensym(Symbol(:val_, symbs[i])) for i in 1:length(prefs)]
#     vec_accesses = struct_ref_exp(rc)
#     assignments = [Expr(:(=), names[i], vec_accesses[i]) for i in 1:length(prefs)]


#     loop_ind = loop_idxs(rc, filled_idxs) # Index to loop over

#     totalname = gensym(:total)

    
#     # Get all ref indices, and check if there is a single one that has multuple indices, for the axis
#     all_ref_indices = ref_indices.(prefs)
#     indexmaps = index_to_axis.(all_ref_indices)
    

#     vecrefs = [wrapF(prefs[i], Expr(:ref, Symbol(:vec, i), ref_indices(prefs[i])...)) for (i, pref) in enumerate(prefs)]
#     reduce_vecrefs = Expr(:block)
#     prodcall = Expr(:call, :*, (wrapF(rc[i], Expr(:ref, Symbol(:vec, i), ref_indices(rc[i])...)) for i in 1:length(vecrefs))...)
#     wrap_prodcall = wrapF(rc, prodcall)
#     push!(reduce_vecrefs.args, Expr(:+=, totalname, wrap_prodcall))

#     collectex = nested_turbo_wrap(
#         reduce_vecrefs, 
#         (:(axes($(Symbol(:vec, index_first_ref(prefs, loop_ind[i_ind]))), $(indexmaps[index_first_ref(prefs, loop_ind[i_ind])][loop_ind[i_ind]]))) 
#             for i_ind in 1:length(loop_ind)) |> collect, 
#         loop_ind)

#     exp = quote 
#         $(unpack_keyword_expr(idxs, :idxs))
#         $(assignments...)
#         $totalname = zero(promote_eltype($(struct_ref_exp(rc)...)))
#         $(collectex)
#         $totalname
#     end
#     return exp
# end

function (::typeof(BlockModel(allPure)))(rc::RefMult, argstype, filled_idxs, precision = nothing, assignments = nothing)
    debugmode = argstype isa NamedTuple
    if debugmode
        println("Allpure blockmodel refmult: $(rc)")
    end

    first = false
    if isnothing_generated(assignments) # Get the assignemnts of the base prefts
        assignments = assignment_map(rc, idxs)
        debugmode && println("\t", "first assignment map")
        first = true
    end
    # debugmode && println("\tassignments: $assignments")
    # debugmode && println("\tNames: ", get_names(assignments))

    filled_idxs = index_names(filled_idxs)
    ref_inds = ref_indices(rc)
    prefs = get_prefs(rc)
    subassignments = submap(assignments, rc)
    names = get_names(subassignments)
    # names_assignments = get_assignments(rc)
    # names = first.(names_assignments)
    # assignments = [Expr(:(=), names[i], last.(names_assignments)[i]) for i in 1:length(names)]

 
    loop_ind = loop_idxs(rc, filled_idxs)
    totalname = gensym(:total)

    if debugmode    
        println("\t", "filled_idxs: $filled_idxs")
        println("\t", "loop_ind: $loop_ind")
    end


    vexp = value_exp(rc, argstype, filled_idxs)
    interpolate!(vexp, names...)

    axes_names = [names[index_first_ref(flatprefs(rc), loop_ind[i])] for i in 1:length(loop_ind)]

    all_ref_indices = ref_indices.(flatprefs(rc))
    indexmaps = index_to_axis.(all_ref_indices)
    axes_indxs = [indexmaps[index_first_ref(flatprefs(rc), loop_ind[i])][loop_ind[i]] for i in 1:length(loop_ind)]

    # axes_exps = [:(axes($(axes_names[i]), $(axes_indxs[i]))) for i in 1:length(loop_ind)]
    axes_exps = [get_axes_exp(assignments, argstype, filled_idxs, loop_ind[i]) for i in 1:length(loop_ind)]
    
    if debugmode
        println("\t", "axes_names: $axes_names")
        println("\t", "axes_indxs: $axes_indxs")
        println("\t", "axes_exps: $axes_exps")
    end

    collectex = nested_turbo_wrap(
        Expr(:(+=), totalname, vexp),
        axes_exps, 
        loop_ind)

    
    exp = quote
        $(first ? unpack_keyword_expr(filled_idxs, :idxs) : nothing)
        $(first ? unique_assignments(assignments) : nothing)
        # $totalname = promote_eltype($(names...))(0)
        $totalname = $(zero_assignment(names...; precision))
        $collectex
        $totalname
    end
    
        
    return exp
end

