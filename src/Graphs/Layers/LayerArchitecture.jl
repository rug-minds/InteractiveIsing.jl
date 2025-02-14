struct LayerMetaData{LT, IndexSet, StateSet}
    l::LT
end

@inline function typemap(t::Type{<:LayerMetaData}, symb)
    if symb == :LT
        return t.parameters[1]
    elseif symb == :IndexSet
        return t.parameters[2]
    elseif symb == :StateSet
        return t.parameters[3]
    end
end

indexset(lt::Type{<:LayerMetaData}) = typemap(lt, :IndexSet)
stateset(lt::Type{<:LayerMetaData}) = typemap(lt, :StateSet)
stateset(lt::LMD) where LMD<:LayerMetaData = stateset(LMD)
statetype(lt::LayerMetaData) = statetype(lt.l)

function LayerMetaData(l::IsingLayer)
    LayerMetaData{typeof(l), graphidxs(l), stateset(l)}(l)
end

struct LayerArchitecture{MDs} 
    metadatas::MDs
end

function GetArchitecture(ls...)
    LayerArchitecture(map(LayerMetaData, ls))
end