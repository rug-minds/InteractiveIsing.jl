"""
From kwargs in a generated body extract the svec of the namedtuple type
i.e. svec((:s1,:s2...), (T1, T2,...))
"""
generated_body_kwargs_nt_svec(kwargs) = kwargs.parameters[4].parameters
generated_body_kwargs_symbols(kwargs) = generated_body_kwargs_nt_svec(kwargs)[1]
generated_body_kwargs_types(kwargs) = tuple(generated_body_kwargs_nt_svec(kwargs)[2].parameters...)
tuple_type(ts::Tuple) = Tuple{ts...}

resolve_typevar(t::TypeVar) = t.ub
resolve_typevar(t) = t
resolve_typevars(ts::Tuple) = map(resolve_typevar, ts)

function method_function(m::Method)
    ftype = Base.unwrap_unionall(m.sig).parameters[1]
    isdefined(ftype, :instance) || error("Method $m does not correspond to a unique callable instance")
    return getfield(ftype, :instance)
end

function method_arg_types(m::Method)
    sig_params = Base.unwrap_unionall(m.sig).parameters
    return Tuple(resolve_typevars(Tuple(sig_params[2:end])))
end

function select_directmethod_method(func::Function, ms::Vector{Method})
    stdin isa Base.TTY || error("DirectMethod found multiple methods for $func. Use DirectMethod(func, methodidx) in non-interactive mode.")

    options = ["[$idx] $m" for (idx, m) in enumerate(ms)]
    keybindings = length(ms) <= 9 ? [Char('0' + idx) for idx in 1:length(ms)] : Char[]
    menu = REPL.TerminalMenus.RadioMenu(options; pagesize=min(length(options), 10), keybindings, warn=false)
    prompt = "DirectMethod found multiple methods for $func. Select a method:"
    selected = REPL.TerminalMenus.request(prompt, menu)
    selected == -1 && error("Method selection cancelled for $func")
    return selected
end

"""
Reference to a single method of a function
    This allows us to dispatch on the signature and argsymbols of the method

This allows us to reason about the method signature and argument names, 
and to call the method directly with the correct arguments and kwargs, 
without having to worry about multiple methods or default arguments.
"""
struct DirectMethod{Argnames, ArgsSig, Kwargnames, KwargsSig, F} <: Function
    func::F
end

numargs(::Union{DirectMethod{Argnames}, Type{<:DirectMethod{Argnames}}}) where Argnames = length(Argnames)
numkwargs(::Union{DirectMethod{Argnames,ArgsSig,Kwargnames}, Type{<:DirectMethod{Argnames,ArgsSig,Kwargnames}}}) where{Argnames, ArgsSig, Kwargnames} = length(Kwargnames)
argsymbols(::Union{DirectMethod{Argnames}, Type{<:DirectMethod{Argnames}}}) where Argnames = Argnames
kwargsymbols(::Union{DirectMethod{Argnames, ArgsSig, Kwargnames, KwargsSig}, Type{<:DirectMethod{Argnames, ArgsSig, Kwargnames, KwargsSig}}}) where {Argnames, ArgsSig, Kwargnames, KwargsSig} = Kwargnames
argtypes(::Union{DirectMethod{Argnames, ArgsSig}, Type{<:DirectMethod{Argnames, ArgsSig}}}) where {Argnames, ArgsSig} = tuple(ArgsSig.parameters...)
kwargtypes(::Union{DirectMethod{Argnames, ArgsSig, Kwargnames, KwargsSig}, Type{<:DirectMethod{Argnames, ArgsSig, Kwargnames, KwargsSig}}}) where {Argnames, ArgsSig, Kwargnames, KwargsSig} = tuple(KwargsSig.parameters...)



function DirectMethod(func::F,m::Method; allowedkwargs = nothing) where {F<:Function}
    args = Symbol.(split(m.slot_syms, "\0")[2:end-1])
    args_sig = tuple_type(method_arg_types(m))
    kwargs_and_types = method_kwarg_types(m)
    kwargs = tuple(keys(kwargs_and_types)...)
    kwargs_sig = tuple_type(tuple(values(kwargs_and_types)...))

    #subtract kwarg decl from args to get only normal args
    args = tuple(args[1:length(args)-length(kwargs)]...)
    if !isnothing(allowedkwargs)
        if !all(kwargs .∈ Ref(allowedkwargs))
            error("Method $m contains keyword arguments that are not in the allowed set: $(setdiff(kwargs, allowedkwargs))")
        end
    end
    return DirectMethod{args, args_sig, kwargs, kwargs_sig, F}(func)
end

function DirectMethod(func::Function; allowedkwargs = nothing)
    ms = collect(methods(func))
    m = argmax(ms) do m
        getproperty(m, :primary_world)
    end
    return DirectMethod(func, m; allowedkwargs)
end

function DirectMethodPick(func::Function; allowedkwargs = nothing)
    ms = collect(methods(func))
    length(ms) == 1 && return DirectMethod(func, 1; allowedkwargs)
    return DirectMethod(func, select_directmethod_method(func, ms); allowedkwargs)
end

DirectMethod(dm::DirectMethod) = dm
DirectMethod(m::Method) = DirectMethod(method_function(m), m)

function DirectMethod(func::Function, methodidx::Integer)
    @assert length(methods(func)) >= methodidx "Function does not have enough methods"
    m = methods(func)[methodidx]
    return DirectMethod(func,m)
end

"""
Passes args and kwargs to the underlying function
Checks that the number and types of args and kwargs match the method signature

For now this might pass to a different method due to the need for default values
"""
@generated function (dm::DirectMethod)(args...; kwargs...)
    @assert length(args) <= numargs(dm) "Too many positional arguments. Expected at most $(numargs(dm)), got $(length(args))"
    @assert length(generated_body_kwargs_symbols(kwargs)) <= numkwargs(dm) "Too many keyword arguments. Expected at most $(numkwargs(dm)), got $(length(kwargs))"

    _argtypes = argtypes(dm)
    @assert all(args .<: _argtypes) "Positional argument types do not match method signature. Expected $(_argtypes), got $(map(typeof, args))"


    _kwargtypes = kwargtypes(dm)

    inputkwargstypes = generated_body_kwargs_types(kwargs)
    @assert all(inputkwargstypes .<: _kwargtypes) "Keyword argument types do not match method signature. Expected $(_kwargtypes), got $(map(typeof, kwargs))"
    inputkwarg_symbols = generated_body_kwargs_symbols(kwargs)
    expected_kwarg_symbols = kwargsymbols(dm)
    @assert all(inputkwarg_symbols .∈ Ref(expected_kwarg_symbols)) "Keyword argument symbols do not match method signature. Expected $(expected_kwarg_symbols), got $(inputkwarg_symbols)"
    return quote
        @inline dm.func(args...; kwargs...)
    end
end

@generated function pass_existing_kwargs(dm::DM; kwargs...) where DM <: DirectMethod
    passing_kwargs = generated_body_kwargs_symbols(kwargs)
    existing_kwargs = kwargsymbols(dm)

    intersection_kwargs = intersect(passing_kwargs, existing_kwargs)

    
    return quote
        (; $(intersection_kwargs...)) = (;kwargs...)
        @inline dm.func(; $(intersection_kwargs...))
    end
end


"""
    method_kwarg_types(m::Method) -> Dict{Symbol, Any}

Return a dictionary mapping each declared keyword argument of `m` to its
annotation. Unannotated keyword arguments map to `Any`.
"""
function method_kwarg_types(m::Method)
    kwarg_names = Base.kwarg_decl(m)
    isempty(kwarg_names) && return Dict{Symbol, Any}()

    ci = Base.uncompressed_ast(m)

    # Old name-based lookup. This is brittle for anonymous functions because
    # the lowered kw helper name is not necessarily derived from `m.name`.
    # helper_ref = findfirst(ci.code) do stmt
    #     stmt isa GlobalRef || return false
    #     helper_name = String(stmt.name)
    #     return startswith(helper_name, "#" * String(m.name) * "#")
    # end
    # isnothing(helper_ref) && error("Could not locate kwarg helper for method $m")
    #
    # helper_global = ci.code[helper_ref]::GlobalRef
    # helper_func = getfield(helper_global.mod, helper_global.name)
    # helper_method = first(methods(helper_func))

    func_type = Base.unwrap_unionall(m.sig).parameters[1]
    helper_method = nothing
    for stmt in ci.code
        stmt isa GlobalRef || continue

        helper_func = try
            getfield(stmt.mod, stmt.name)
        catch
            continue
        end
        helper_func isa Function || continue

        for candidate in methods(helper_func)
            helper_sig = Base.unwrap_unionall(candidate.sig).parameters
            length(helper_sig) >= 2 + length(kwarg_names) || continue
            helper_sig[2 + length(kwarg_names)] == func_type || continue
            helper_method = candidate
            break
        end
        !isnothing(helper_method) && break
    end
    isnothing(helper_method) && error("Could not locate kwarg helper for method $m")

    helper_sig = Base.unwrap_unionall(helper_method.sig).parameters
    kwarg_types = Tuple(resolve_typevars(Tuple(helper_sig[2:1 + length(kwarg_names)])))

    return Dict{Symbol, Any}(name => type for (name, type) in zip(kwarg_names, kwarg_types))
end
