using InteractiveIsing, Processes, LoopVectorization, BenchmarkTools, SparseArrays, JET
import InteractiveIsing as II

g = IsingGraph(30,30,30, type = Discrete, periodic = (:z, :y))
createProcess(g)
w = @WG "dr -> 1/dr" NN=1
genAdj!(g[1], w)
pause(g)


resize!(getparams(g,:self).val, length(state(g)))
setparam!(g, :self, rand(Float32, length(state(g))), true)
setparam!(g, :b, rand(Float32, length(state(g))), true)

# function test1(v1,v2)
#     total = zero(eltype(v1))
#     @turbo for i in eachindex(v1)
#         total += v1[i]
#     end
#     @turbo for i in eachindex(v2)
#         total += v2[i]
#     end
#     return total
# end

# function testrc(as; j = 1)
#     adj = as.gadj
#     cumsum = zero(eltype(as.gstate))
#     @turbo for ptr in nzrange(adj, j)
#         i = adj.rowval[ptr]
#         wij = adj.nzval[ptr]
#         cumsum += wij * as.gstate[i] 
#     end
#     return cumsum
# end

function testturbo(v)
    t = 0
    @turbo for i in eachindex(v)
        t += v[i]
    end
    return t
end


g.hamiltonian = NIsing(g)

as = prepare(II.MetropolisNew(), (;g))

as = (;as..., newstate = II.SparseVal(-as.gstate[1], 1, length(as.gstate)))
as.gstate .= (rand(length(as.gstate)) ./ 10)
# as.gstate .= 1

si = @ParameterRefs s_i
sid2 = @ParameterRefs s_i/2
snid2 = @ParameterRefs sn_i/2
si2 = @ParameterRefs s_i^2

bi = @ParameterRefs b_i
bi2 = @ParameterRefs b_i^2
# get(bi, as).val .= rand(length(get(bi, as).val))


rr1 = @ParameterRefs s_i+b_i
rr25 = @ParameterRefs s_i+b_i-sn_j
rr2 = @ParameterRefs s_i+b_i^2-sn_j/2
# @benchmark rr2($as)
rr2(as)
function rr2test(si, bi, snj)
    total = 0f0
    @turbo for i in eachindex(si)
        total += si[i]
    end
    @turbo for i in eachindex(bi)
        total += (bi[i])^2
    end
    total -= II.nzval(snj)/2

    return total
end

gsi = get(si, as)
gsi64 = Float64.(gsi)
gbi = get(bi, as)
gbi2 = get(bi2, as)
gsn = get(snid2, as)
# rr2test(gsi, gbi2, gsnid2)

@benchmark rr2test($gsi, $gbi2, $gsn)

@benchmark $rr2($as)


rr3 = @ParameterRefs (s_j - sn_j)^2 + b_i

# rc1 = @ParameterRefs (sn_i - s_i)*b_i

# function rc1test(si, sj)
#     total = 0f0
#     for i in eachindex(si)
#         for j in eachindex(sj)
#             total += si[i]*sj[j]
#         end
#     end
#     return total
# end

# function rc1testturbo(si, sj)
#     total = 0f0
#     @turbo for i in eachindex(si)
#         for j in eachindex(sj)
#             total += si[i]*sj[j]
#         end
#     end
#     return total
# end

# rc1test(gsi, gsi)
# rc1testturbo(gsi, gsi)
# # @benchmark rc1test($gsi, $gsi)
# # @benchmark rc1testturbo($gsi, $gsi)

# rc2 = @ParameterRefs s_i*s_i + sn_i

# function rc2test(si, sni)
#     total = 0f0
#     @turbo for i in eachindex(si)
#         for j in eachindex(si)
#             total += si[i]^2
#         end
#     end
#     return total + sni[]
# end

# rc3 = @ParameterRefs s_i*b_i


function sntestturbo(si, sn)
    total = 0f0
    @turbo for j in axes(si, 1)
            total += (^)(si[j] + -sn[j], 2)
    end
    return total
end

function sntestsimd(si, sn)
    total = 0f0
    @fastmath @simd for j in axes(si, 1)
            total += (^)(si[j] + -sn[j], 2)
    end
    return total
end

function sntest_factorized(si, sn)
    total = 0f0
    @turbo for j in axes(si, 1)
            total += si[j]
    end
    total += II.nzval(sn)^2 -2*II.nzval(sn)*si[nzrange(sn)]
    return total
end



# v1 = rand(1000)
# v2 = rand(1000)

# @benchmark sntestturbo($gsi, $gsn)
# @benchmark sntestsimd($gsi, $gsn)
# @benchmark sntest_factorized($gsi, $gsn)


# MULTS

rm1 = @ParameterRefs s_i*b_i
rm2 = @ParameterRefs s_i*b_i*sn_j
rmsp = @ParameterRefs (s_i*w_ij)*(sn_j-s_j) + b_i/2 + b_i
swij = @ParameterRefs s_i*w_ij

sswij = @ParameterRefs (s_i*w_ij)*s_j

a = @ParameterRefs a_



function another_ttest(v, r)
    total = 0f0
    val = r[]
    @turbo for i in eachindex(v)
        total += v[i]/val
    end
    return total
end
v = rand(Float32, 1000)
r = Ref(1f0)

# @benchmark another_ttest($v, $r)


function vmtest(v, m)
    total = 0f0
    j = 3
    @turbo for i in eachindex(v)
        total += v[i]*m[i,j]
    end
    return total
end

v = rand(Float32, 1000)
m = rand(Float32, 1000, 1000)

# @benchmark vmtest($v, $m)


h = g.hamiltonian + DepolField(g)
g.hamiltonian = h
as = (;as..., hamiltonian = h)
dpp = II.deltaH(II.hamiltonians(h)[3])
fulldh = II.deltaH(h)
f1 = fulldh[1]
f2 = fulldh[2]
f3 = fulldh[3]
f4 = fulldh[4]


function testias(args, idxs)
     #= /Users/fabian/Documents/GitHub/InteractiveIsing.jl/src/Utils/Parameters/ParameterRefs/RefMult.jl:340 =#
     (; j) = idxs
     #= /Users/fabian/Documents/GitHub/InteractiveIsing.jl/src/Utils/Parameters/ParameterRefs/RefMult.jl:341 =#
     var"##dpf#433" = getindex(getproperty(getproperty(args, :hamiltonian), :dpf))
     var"##c#434" = getindex(getproperty(getproperty(args, :hamiltonian), :c))
     var"##s#435" = getindex(getproperty(args, :gstate), j)
     var"##sn#436" = getindex(getproperty(args, :newstate))
     #= /Users/fabian/Documents/GitHub/InteractiveIsing.jl/src/Utils/Parameters/ParameterRefs/RefMult.jl:342 =#
     begin
         #= /Users/fabian/Documents/GitHub/InteractiveIsing.jl/src/Utils/Parameters/ParameterRefs/BlockModels.jl:268 =#
         nothing
         #= /Users/fabian/Documents/GitHub/InteractiveIsing.jl/src/Utils/Parameters/ParameterRefs/BlockModels.jl:269 =#
         nothing
         type = @inline II.promote_eltype(var"##dpf#433", var"##c#434", var"##s#435", var"##sn#436")
        #  println(type)
         #= /Users/fabian/Documents/GitHub/InteractiveIsing.jl/src/Utils/Parameters/ParameterRefs/BlockModels.jl:271 =#
         var"##total#437" = type(0)
        # var"##total#437" = 0f0

         #= /Users/fabian/Documents/GitHub/InteractiveIsing.jl/src/Utils/Parameters/ParameterRefs/BlockModels.jl:272 =#
         var"##total#437" = (var"##dpf#433" / var"##c#434") * (var"##s#435" - var"##sn#436")
         #= /Users/fabian/Documents/GitHub/InteractiveIsing.jl/src/Utils/Parameters/ParameterRefs/BlockModels.jl:273 =#
         var"##total#437"
     end
end
testias(as, (;j = 1))

@benchmark testias(as, (;j = 1))



