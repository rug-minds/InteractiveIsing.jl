struct Test{T} <: AbstractVector{T}
    v::Vector{T}
end

Base.size(t::Test) = size(t.v)
Base.getindex(t::Test, idx::Integer) = t.v[idx]
Base.setindex!(t::Test, value, idx::Integer) = t.v[idx] = value

struct DTest{K,V} <: AbstractDict{K,V}
    v::Vector{V}

    # function DTest(pairs::Pair{K,V}...) where {K,V}
    #     d = new{K,V}(Vector{V}(undef,length(pairs)))
    #     for (idx, p) in enumerate(pairs)
    #         d.v[idx] = p.second
    #     end
    #     return d
    # end

    function DTest(v::Vector{V}) where V
        return new{Int,V}(v)
    end
end

Base.keys(d::DTest) = String.(collect(1:length(d.v)))
Base.values(d::DTest) = d.v
Base.getindex(d::DTest, key) = d.v[parse(Int, String(key))]