export PhysicalWeightGenerator

"""
    PhysicalWeightGenerator(generator, scales; units = physicalunits(energy = 1))

Wrap a weight generator so `dr` is passed as a Unitful length and Unitful
returned weights are converted to internal coupling values at adjacency-build
time.
"""
struct PhysicalWeightGenerator{WG,S,U} <: AbstractWeightGenerator
    generator::WG
    scales::S
    units::U
end

function PhysicalWeightGenerator(generator::WG, scales = nothing; units = physicalunits(energy = 1)) where {WG<:AbstractWeightGenerator}
    return PhysicalWeightGenerator{WG,typeof(scales),typeof(units)}(generator, scales, units)
end

getNN(wg::PhysicalWeightGenerator) = getNN(wg.generator)
getNN(wg::PhysicalWeightGenerator, dims) = getNN(wg.generator, dims)

"""
    _bind_physical_weight_scales(weightgen, physical_scales)

Attach graph or layer physical scales to a physical weight generator that was
created without an explicit scale context. Explicit generator scales win.
"""
function _bind_physical_weight_scales(wg::PhysicalWeightGenerator{WG,Nothing,U}, physical_scales) where {WG<:AbstractWeightGenerator,U}
    return PhysicalWeightGenerator(wg.generator, physicalscales(physical_scales); units = wg.units)
end

function _bind_physical_weight_scales(wg::WG, physical_scales::S) where {WG,S}
    return wg
end

@inline function getWeight(wg::PhysicalWeightGenerator{WG,S,U}; dr::DR, c1::C1 = nothing, c2::C2 = nothing, dc::DC = nothing) where {WG<:AbstractWeightGenerator,S,U,DR,C1,C2,DC}
    scales = physicalscales(wg.scales)
    isnothing(scales.length[]) && throw(MissingPhysicalScale(:J, :length, dr))
    physical_dr = dr * scales.length[]
    raw = getWeight(wg.generator; dr = physical_dr, c1 = c1, c2 = c2, dc = dc)
    return internalvalue(raw, wg.units, scales, nothing; parameter = :J)
end

@inline function (wg::PhysicalWeightGenerator; dr::DR, c1::C1 = nothing, c2::C2 = nothing, dc::DC = nothing) where {DR,C1,C2,DC}
    return getWeight(wg; dr = dr, c1 = c1, c2 = c2, dc = dc)
end
