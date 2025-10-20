using InteractiveIsing
import InteractiveIsing as II

g = IsingGraph(30,30,30, stype = Continuous, periodic = (:z, :y))

@ParameterRefs function test_paramref(a,b)
    return a[i,j]*b[j]
end