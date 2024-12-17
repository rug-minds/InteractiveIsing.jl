export @CompositeAlgorithm

struct TriggerList
    triggers::Vector{Int}
    idx::Int
end

TriggerList() = TriggerList([], 1)

TriggerList(v::Vector{Int}) = TriggerList(v, 1)

next(tl::TriggerList) = tl.triggers[tl.idx]
inc!(tl::TriggerList) = tl.idx += 1

abstract type CompositeAlgorithm end

macro CompositeAlgorithm(name, funcs_n_intervals...)
    println("Name: $name")
    println("Funcs_n_intervals: $funcs_n_intervals")
    # Funcs are uneven
    nfuncs = findfirst(x -> x isa Int, funcs_n_intervals) - 1
    println("Nfuncs: $nfuncs")
    nintervals = length(funcs_n_intervals) - nfuncs
    funcs = funcs_n_intervals[1:nfuncs]
    intervals = funcs_n_intervals[nfuncs+1:end]

    struct_def = quote 
        struct $name <: CompositeAlgorithm end
    end

    underscore_name = Symbol("_$name")

    method_defs = quote
        function $name(args)
            (;runtime) = args
            $underscore_name(args, runtime)
        end

        function $underscore_name(args, ::Indefinite)
            $(indefinite_composite_body(funcs, intervals))
        end

        function $underscore_name(args, ::Repeat{repeats}) where repeats
            $(repeat_composite_body(funcs, Val.(intervals)))
        end

        function prepare(::$name, args)
            # $(make_composite_prepare(funcs, intervals))
            @$
        end
    end

    prep = quote 
        function prepare(::$name, args)
            @$
        end 
    end

    # interpolate!(method_defs, indefinite_composite_body(funcs, intervals), repeat_composite_body(funcs, Val.(intervals)))
    interpolate!(prep, make_composite_prepare(funcs, intervals))
    remove_line_number_nodes!(method_defs)
    remove_line_number_nodes!(prep)
    println(method_defs)
    println(prep)
end

function indefinite_composite_body(funcnames, intervals)
    ifs =(
            :(
                if loopidx(proc) % $(intervals[i]) == 0
                    $(funcnames[i])(args)
                end
            )
            for i in 1:length(funcnames)
        )

    expcat(ifs...) |> remove_line_number_nodes
end

# Uses precomputed triggers
function repeat_composite_body(funcnames, intervals)
    
    ifs = (
        part_composite_exp_iterated(funcnames[i], i, intervals[i])
        for i in 1:length(funcnames)
    )
    expcat(:((;triggers) = args), ifs...) |> remove_line_number_nodes
end

function make_composite_prepare(funcnames, intervals)
    precompute_triggers = quote
        intervals = $intervals
        (;runtime) = args
        if runtime <: Repeats
            _repeats = repeats(runtime)
            Triggers = ((TriggerList() for _ in eachindex($funcnames))...,)
            for i in 1:_repeats
                for (i_idx, interval) in enumerate(intervals)
                    if i % interval == 0
                        push!(Triggers[i_idx].triggers, i)
                    end
                end
            end 
        end
    end

    expcat(precompute_triggers, (:(prepare($func, args)) for func in funcnames)...) |> remove_line_number_nodes
end

function part_composite_exp_repeat(funcname, ::Val{N}) where N
    if N == 1
        return :($funcname(args))
    else
        return :(if loopidx(proc) % $N == 0
                    $funcname(args)
                end)
    end
end

function part_composite_exp_iterated(funcname, idx, ::Val{N}) where N
    if N == 1
        return :($funcname(args))
    else
        return :(if loopidx(proc) == next()
                    $funcname(args)
                    inc!(triggers[$idx])
                end)
    end
end

