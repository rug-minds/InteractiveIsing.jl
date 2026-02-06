using InteractiveIsing
import InteractiveIsing: Parameters, ParamTensor, GeneratedParameters
p1 = ParamTensor([1:10;], 0, "", true)
p2 = ParamTensor([1:1.:10;], 1, "", false)
p20 = ParamTensor([1:1.:10;], 0, "", false)
p3 = ParamTensor(10, 1, "", false)

const i1 = Parameters(;p1 = deepcopy(p1), p2 = deepcopy(p2), p3= deepcopy(p3))
const i2 = Parameters(;p1 = deepcopy(p1), p2 = deepcopy(ParamTensor(p2, true)), p3 = deepcopy(p3))
const i3 = Parameters(;p1 = deepcopy(p1), p2 = deepcopy(p20), p3 = deepcopy(p3))

# function test2(params::Parameters)
#     cum = eltype(params.p1)(0)
#     param1 = params.p1
#     param2 = params.p2
#     @turbo for idx in 1:length(param1)
#         p1 = param1[idx]
#         p2 = param2[idx]
#         cum += p1 + p2
#     end
#     return cum
# end 

@GeneratedParameters function test3(params::Parameters)
    cum = 0.
    @turbo for idx in eachindex(params.p1)
        cum += params.p1[idx] + params.p2[idx]
    end
    return cum
end

test3_exp(i1)