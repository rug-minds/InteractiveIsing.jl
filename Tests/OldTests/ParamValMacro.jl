using MacroTools, InteractiveIsing
import InteractiveIsing: Parameters
function find_type_in_args(args, type::Symbol)
    for arg in args
        if arg isa Expr && arg.head == :(::)
            if  arg.args[2] == type
                return arg.args[1]
            end
        end
    end
    return nothing
end

function get_field_type(::Type{T}, field::Val{S}) where {T <: NamedTuple,S}
    names = fieldnames(T)
    types = fieldtypes(T)
    
    index = findfirst(==(S), names)
    
    return index === nothing ? throw(ArgumentError("Field $field not found")) : types[index]
end

function get_field_type(@specialize(nt::NamedTuple), field)
    return get_field_type(typeof(nt), field)
end

get_field_type(nt, field::Symbol) = get_field_type(nt, Val{field}())

get_field_type(ip::Parameters{nt}, field::Symbol) where nt = get_field_type(nt, field)

function replace_struct_field_access(func, paramname, replace)
    find_struct_accesses = find_struct_field_access(func, paramname)
    quotes = []
    places = []
    for access in find_struct_accesses
        idxs = access[1]
        fieldname = access[2]
        q = replace(func, paramname, fieldname, idxs)
        push!(quotes, q)
        push!(places, idxs[1:end-2])
    end
    for (qidx,q) in enumerate(quotes)
        func = replace_symb(func, q, places[qidx])
    end
    return func
end

function replace_paramtensor(func, paramname)
    function replacefunc(containing_exp, paramname, fieldname) 
        containing_exp = enter_args(func, idxs[1:end-2])
        if containing_exp.head == :ref
            replaced = :(default($paramname, $fieldname))
        else
            replaced = replace_symb(containing_exp, :(default($paramname, $fieldname)), idxs[end-1])
        end
        q = quote   if isactive($paramname, $(fieldname))
                    $(containing_exp)
                else
                    $(replaced)
                end
        end
        return q
    end
    replace_struct_field_access(func, paramname, replacefunc)
end
using LoopVectorization

"""
Find all places in an expression where we have str.field or str.field[idx]
"""
function find_struct_access(exp, symb)
    find_idxs = find_symbs(exp, symb)
    deletes = []
    for i_idx in eachindex(find_idxs)
        this_idxs = find_idxs[i_idx][1:end-1]
        new_idxs = this_idxs
        head = get_head(exp, this_idxs)
        if head != :. && head != :ref
            push!(deletes, i_idx)
            continue
        end
        while (head == :ref || head == :.)
            new_idxs = new_idxs[1:end-1]
            head = get_head(exp, new_idxs)
        end
        find_idxs[i_idx] = find_idxs[i_idx][1:(length(new_idxs)+1)]
    end
    deleteat!(find_idxs, deletes)

    return find_idxs
end

"""
Replace the args of an expression at level idxs with replace
"""
function replace_args!(exp, idxs, replace)
    enter_args(exp, idxs[1:end-1]).args[idxs[end]] = replace
end

"""
Replace all struct accesses of a symbol in an expression with a new expression
    Give a function to replace the expression in the form replace(this_exp)
"""
function replace_struct_access(exp, symb, replace)
    find_idxs = find_struct_access(exp, symb)
    for i_idx in eachindex(find_idxs)
        this_idxs = find_idxs[i_idx]
        this_exp = enter_args(exp, this_idxs)
        this_replace = if replace isa Function
                        replace(this_exp)
                    else
                        replace
                    end
            
        replace_args!(exp, this_idxs, this_replace)
    end
end

function structname_fieldname(struct_access_exp)
    found = (expmatch(struct_access_exp) do head, args
        if head == :.
            return true
        end
        return false
    end)
    structname = found[1].args[1]
    fieldname = found[1].args[2]
    return structname, fieldname
end

function default_paramtensor(paramname, fieldname)
    :(default($paramname, $fieldname))
end

function ifactive_defaultparamtensor(exp, paramname, fieldname)
    quote   if isactive($paramname, $(fieldname))
                $exp
            else
                default($paramname, $fieldname)
            end
    end
end

function arguments_to_pass(a...)
    new_a = []
    for arg in a
        if arg isa Expr && arg.head == :(::)
            push!(new_a, arg.args[1])
        else
            push!(new_a, arg)
        end
    end
    return new_a
end

function inline_default(exp, params::Union{Parameters, Type{<:Parameters}}, fieldname)
    # fieldname = Symbol(fieldname)
    if fieldname isa QuoteNode
        fieldname = fieldname.value
    end

    return if isactive(params, fieldname)
        exp
    else
        default(params, fieldname)
    end
end


macro ParamTensorFunc(func)
    # println("Func ", func)
    @capture(func, function name_(a__) body__ end)
    paramname = find_type_in_args(a, :Parameters)
    replace_struct_access(func, paramname, (exp) -> ifactive_defaultparamtensor(exp, structname_fieldname(exp)...))
    println(func)
    return esc(func)
end

@ParamTensorFunc function test2(params::Parameters)
    cum = paramzero(params.p1)
    @simd for idx in 1:length(params.p1)
        p1 = params.p1[idx]
        p2 = params.p2[idx]
        cum += p1 + p2
    end
    return cum
end

macro GeneratedParamTensor(func)
    @capture(func, function fname_(a__) body_ end)
    paramname = find_type_in_args(a, :Parameters)
    find_struct_accesses = find_struct_field_access(func, paramname)
    include_or_not = []
    include_names = []
    for access in find_struct_accesses
        idxs = access[1]
        fieldname = access[2]
        includename = Symbol(:include_, :($fieldname))
        push!(include_names, includename)
        push!(include_or_not, quote $includename = isactive($paramname, $fieldname) ? true : false end)
    end

    wrappedbody = Expr(:quote, body)
    wrappedparamname = Expr(:quote, paramname)
    args2pass = arguments_to_pass(a...)
    expfunc = 
        quote 
            # Create the expression function for the generated function
            function $(Symbol(:($fname), :_exp))($(args2pass...))
                rawbody = $wrappedbody
                paramname = $wrappedparamname

                # Inline the default values
                replace_struct_access(rawbody, paramname, (exp) -> inline_default(exp, $(paramname),structname_fieldname(exp)[2]))
                return rawbody
            end
        end
    realfunc =
        quote
            @generated function $fname($(a...))
                # Return the expression function
                return $(Symbol(:($fname), :_exp))($(args2pass...))
            end
        end

    returnexp = quote
        $expfunc
        $realfunc
    end
 
    println(a)
    println("Return exp: ", returnexp)
    return esc(returnexp)
    # return nothing
end
@GeneratedParamTensor function testtgen(idx, params::Parameters)
    return params.p1[idx] + params.p2[idx]
end

p1 = ParamTensor([1:10;], 0, "", true)
p2 = ParamTensor([1:1.:10;], 1, "", false)
p20 = ParamTensor([1:1.:10;], 0, "", false)
p3 = ParamTensor(10, 1, "", false)

const i1 = Parameters(;p1 = deepcopy(p1), p2 = deepcopy(p2), p3= deepcopy(p3))
const i2 = Parameters(;p1 = deepcopy(p1), p2 = deepcopy(ParamTensor(p2, true)), p3 = deepcopy(p3))
const i3 = Parameters(;p1 = deepcopy(p1), p2 = deepcopy(p20), p3 = deepcopy(p3))

testtgen_exp(1, i1)


function test2(params::Parameters)
    cum = eltype(params.p1)(0)
    param1 = params.p1
    param2 = params.p2
    @turbo for idx in 1:length(param1)
        p1 = param1[idx]
        p2 = param2[idx]
        cum += p1 + p2
    end
    return cum
end 

@GeneratedParamTensor function test3(params::Parameters)
    cum = 0.
    @turbo for idx in eachindex(params.p1)
        cum += params.p1[idx] + params.p2[idx]
    end
    return cum
end

function test3S(param1, param2)
    cum = 0.
    @turbo for idx in eachindex(param1)
        p1 = param1.val[idx]
        p2 = param2.val[idx]
        cum += p1 + p2
    end
    return cum
end

using BenchmarkTools
# @benchmark test3S(i1.p1, i1.p2)