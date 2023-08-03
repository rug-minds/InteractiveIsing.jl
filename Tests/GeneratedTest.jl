using Random
abstract type GenerationType end
struct Continuous <: GenerationType end
struct Discrete <: GenerationType end

struct GRange{Range, GT <: GenerationType}  end

const range1 = (-1f0,1f0)
const range2 = (0f0,1f0)
const grange1D = GRange{range1, Discrete}()
const grange1C = GRange{range1, Continuous}()
const grange2D = GRange{range2, Discrete}()
const grange2C = GRange{range2, Continuous}()

@inline @generated function gcollectRands(rng, grange::GRange{Range, GType}) where {Range, GType <: GenerationType}
    r_begin = Range[1]
    r_end = Range[2]
    range_len = r_end - r_begin

    if GType == Continuous
        return Meta.parse("return $(range_len)f0*rand(rng, Float32) + $(r_begin)f0")
    else
        return Meta.parse("return rand(rng, $Range)")
    end
end

@inline function collectRands(rng, rnge::Tuple, ::Type{Continuous})
    r_begin = rnge[1]
    r_end = rnge[2]
    range_len = r_end - r_begin

    return range_len*rand(rng, Float32) + r_begin
  
end

@inline function collectRands(rng, rnge::Tuple, ::Type{Discrete})
    return rand(rng, rnge)
end

function repeatGFunc(func, GRange::GR,  n) where {GR <: GRange}
    rng = MersenneTwister(1234)
    cumsum = 0.f0
    ti = time()
    for _ in 1:n
        cumsum += func(rng, GRange)
    end
    println("Time: $(time()-ti)")
    return cumsum
end

function repeatFunc(tp::Type{GT}, func, rnge::Tuple, n) where {GT <: GenerationType}
    rng = MersenneTwister(1234)
    cumsum = 0.f0
    ti = time()
    for _ in 1:n
        cumsum += func(rng, rnge, tp)
    end
    println("Time: $(time()-ti)")
    return cumsum
end

const repeats = 10^8
repeatGFunc(gcollectRands, grange1D, repeats)
repeatFunc(Discrete, collectRands, range1, repeats)

repeatGFunc(gcollectRands, grange1C, repeats)
repeatFunc(Continuous, collectRands, range1, repeats)

# @time repeatGFunc(gcollectRands, grange2D, repeats)
# repeatGFunc(gcollectRands, grange2C, repeats)
# @time repeatGFunc(grange2C, repeats)


# @time repeatFunc(Discrete, collectRands, range2, repeats)
repeatFunc(Continuous, collectRands, range2, repeats)
# @time repeatFunc(Continuous, range2, repeats)



function repeatFunc(tp::GT, rnge::Tuple, n) where {GT <: GenerationType}
    rng = MersenneTwister(1234)
    cumsum = 0.f0
    for _ in 1:n
        cumsum += collectRands(rng, rnge, tp)
    end
    return cumsum
end

function repeatGFunc(GRange::GR,  n) where {GR <: GRange}
    rng = MersenneTwister(1234)
    cumsum = 0.f0
    for _ in 1:n
        cumsum += gcollectRands(rng, GRange)
    end
    return cumsum
end