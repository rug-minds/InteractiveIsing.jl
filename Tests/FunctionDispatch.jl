
const function_registry = Dict{Type{<:Function}, Function}()

register_function(f::Function) = function_registry[typeof(f)] = f
get_registered_function(f::Type{<:Function}) = function_registry[f]

struct CompositeAlgo{FTs, intervals}
    fs::FTs
end

get_function_types(::Type{CompositeAlgo{FTs, intervals}}) where {FTs, intervals} = FTs
get_intervals(::Type{CompositeAlgo{FTs, intervals}}) where {FTs, intervals} = intervals


function CompositeAlgo(fs::FTs, intervals::NTuple{N,Int}) where {N,FTs <: NTuple{N,Function}}
    for f in fs
        register_function(f)
    end
    CompositeAlgo{FTs, intervals}(fs)
end

function test1(a,b)
    return a+b
end

function test2(a,b)
    return a*b
end

function test3(a,b)
    return a-b
end

ca = CompositeAlgo((test1, test2, test3), (1, 2, 3))

function loopexp(runtime, ca::Type{<:CompositeAlgo})
    q = quote end
    for (fidx, functype) in enumerate(get_function_types(ca).parameters)
        f = get_registered_function(functype)
        interval = get_intervals(ca)
        push!(q.args, generate_intervalled_algo(f, interval[fidx]))
    end
    return q
end


function generate_intervalled_algo(f, interval)
    if interval != 1
        return quote
            if loopidx % $interval == 0
                $f(args)
            end
        end
    else
        return quote
            $f(args)
        end
    end
end

loopexp(1000, typeof(ca))