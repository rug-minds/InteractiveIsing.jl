## PARTIAL FUNCTION APPLICATIONS
export PartialF
struct PartialF{F, Args, NArgs} <: Function end

function PartialF(f::Function, args...) 
    # Count nothings in args
    n_nothing = count(isnothing, args)
    return PartialF{f, args, n_nothing}()
end

getF(pf::PartialF{F, Args, NArgs}) where {F, Args, NArgs} = F
getF(pf::Type{<:PartialF}) = getF(pf())
nargs(pf::PartialF{F, Args, NArgs}) where {F, Args, NArgs} = NArgs
nargs(pf::Type{<:PartialF}) = nargs(pf())
fixed_args(pf::PartialF{F, Args, NArgs}) where {F, Args, NArgs} = Args
fixed_args(pf::Type{<:PartialF}) = fixed_args(pf())


fsymbol(f::Function) = nameof(f)
fsymbol(f::Type{<:Function}) = Symbol(string(nameof(f))[2:end])

"""
Creates an expression for a function application on the given arguments
"""
function partialf_exp(pf::Union{Type{<:Function}, Function}, argnames...)
    f = fsymbol(pf)
    if f == :identity && length(argnames) == 1
        return :($(argnames[1]))
    end
    q = :($f($(argnames...)))
    remove_line_number_nodes!(q)
    return q
end

"""
Generate an expression for a partial function application
    pf: the partial function
    argnames: the names of the arguments to the partial function
        i.e. the ones that where given as nothing
    Generates something like:
        F(a, nothing, b, nothing, c) -> F(a, argnames[1], b, argnames[2], c)
"""
function partialf_exp(pf::Union{Type{<:PartialF}, PartialF}, argnames...)
    n_a = 1
    fargs = fixed_args(pf)

    as = nothing
    if !isempty(argnames)
        as = ntuple(i -> isnothing(fargs[i]) ? let idx = n_a; n_a += 1; :($(argnames[idx])) end : :($(fargs[i])), length(fargs))
    else
        as = ntuple(i -> isnothing(fargs[i]) ? let idx = n_a; n_a += 1; :(args[$idx]) end : :($(fargs[i])), length(fargs))
    end
    F = getF(pf)
    # Apply the args
    return :($F($(as...)))
end


@generated function (pf::PartialF{F, Args, NArgs})(args::Vararg{Any, NArgs}) where {F, Args, NArgs}
    return partialf_exp(pf)
end