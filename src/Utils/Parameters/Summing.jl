const idxnames = (:i, :j, :k, :l, :m, :n)

getkeys(idxs) = (idxs isa Tuple || idxs <: Tuple) ? idxs : index_names(idxs)

function dimsum_common_exp(idxs, N; fexp = (:(F(@$))), valname = :pv, idxsname = :idxs, precision = nothing)
    totalname = gensym(:total)
    filled_inds = getkeys(idxs)

    zeroval = isnothing(precision) ? :(eltype($valname)(0)) : :(zero($precision))

    vecname = gensym(:vec)
    looping_idxs = idx_subtract(idxnames[1:N], filled_inds)
    
    idx_assignment = :()
    if !isempty(idxs)
        idx_assignment = quote
            ($(filled_inds)...) = idxs
        end
    end


    fcall = interpolate(fexp,  Expr(:ref, vecname, idxnames[1:N]...))
    add_to_total = Expr(:+=, totalname, fcall)
    add_to_total = nested_turbo_wrap(add_to_total, (:(axes($vecname, $i_ind)) for i_ind in 1:length(looping_idxs)) |> collect, looping_idxs)
    
    
    if !isempty(idxs)
        return quote
            (;$(filled_inds...)) = $idxsname
            $vecname = $valname
            $totalname = eltype($vecname)(0)
            $add_to_total
            $totalname
        end
    else
        return quote
            $vecname = $valname
            $totalname = eltype($vecname)(0)
            $add_to_total
            $totalname
        end
    end
end

function dimsum_exp(pv::Union{Type{<:ParamVal}, ParamVal}, idxs::Union{Type{<:NamedTuple}, NamedTuple, Tuple} = @NamedTuple{}; fexp = (:(F(@$))), valname = :pv, idxsname = :idxs)
    if !isactive(pv)
        return :(length(pv)*default(pv))
    elseif  homogenousval(pv)
        return :((length(pv))*(pv[1]))
    end

    N = dims(pv)
    return dimsum_common_exp(idxs, N; fexp, valname, idxsname)
end

function dimsum_exp(pv::Union{Type{<:AbstractArray{T,N}}, AbstractArray{T,N}}, idxs::Union{Type{<:NamedTuple}, NamedTuple, Tuple} = @NamedTuple{}; fexp = (:(F(@$))), valname = :pv, idxsname = :idxs) where {T,N}
    return dimsum_common_exp(idxs, N; fexp, valname, idxsname)
end

function dimsum_exp(sp::Union{Type{<:SparseVal}, SparseVal}, idxs::Union{Type{<:NamedTuple}, NamedTuple, Tuple} = @NamedTuple{}; fexp = (:(F(@$))), valname = :pv, idxsname = :idxs)
    idxs = getkeys(idxs)
    if :i ∈ idxs
        val = :($valname[$i])
    else
        val = :(nzval($valname))
    end
    return interpolate(fexp, val)
end

@generated function dimsum(pv::ParamVal, idxs = (;), F = identity)
    return dimsum_exp(pv, idxs)
end






