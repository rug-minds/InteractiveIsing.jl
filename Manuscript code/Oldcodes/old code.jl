#=
Old Route-style reference block.
Kept only as a syntax reference; the active code above uses the DSL version.

Metro_T_route = CompositeAlgorithm(dynamics, M_Integrator, M_Logger, B_Logger, T_Logger,
    (1, 1, point_repeat, point_repeat, point_repeat),
    Route(dynamics => M_Integrator, :proposal => :Δvalue,
        transform = accepted_proposal_delta_base),
    Route(M_Integrator => M_Logger, :total => :value),
    Route(dynamics => B_Logger, :hamiltonian => :value, transform = x -> x.b[]),
    Route(dynamics => T_Logger, :model => :value, transform = temp)
)

anneal_partB_route = CompositeAlgorithm(Metro_T_route, AnealingB,
    (1, point_repeat),
    Route(dynamics => AnealingB, :model),
)
Anealing_step_route = Routine(anneal_partB_route, (anneal_time,))

Metro_Pulse_route = CompositeAlgorithm(dynamics, M_Integrator, M_Logger, B_Logger,
    (1, 1, point_repeat, point_repeat),
    Route(dynamics => M_Integrator, :proposal => :Δvalue,
        transform = accepted_proposal_delta_base),
    Route(M_Integrator => M_Logger, :total => :value),
    Route(dynamics => B_Logger, :hamiltonian => :value,
        transform = x -> x.b[]),
)
pulse_part1_route = CompositeAlgorithm(Metro_Pulse_route, pulse1, Graph_Logger,
    (1, point_repeat, capture_interval1),
    Route(dynamics => Graph_Logger, :model => :array, transform = state)
)
relax_part1_route = CompositeAlgorithm(Metro_Pulse_route, Graph_Logger,
    (1, capture_interval2),
    Route(dynamics => Graph_Logger, :model => :array, transform = state)
)
Pulse_and_Relax_route = Routine(pulse_part1_route, relax_part1_route,
    (pulse_time, relax_time),
    Route(dynamics => pulse1, :hamiltonian, :M),
)
=#


#### 可以给每一个landau项写个vector，实现不同的dipole有不同的local potential
#### StateLike可以是OffsetArray, UniformArray, ConstFill，（ConstValue = ConstFill with dimension 0）

#=
现在我们可以分别使用c,localpotential 在ising，quartic，sextic里设置不同参数。
    如果使用默认数值，那多次项会和Jii耦合在一起。如果Jii是2， 后面的多次项相当于都含有一个2.
    调整参数的时候要注意
=#
#=
如果使用这个方式，Ising项的c参数和localpotential项的参数就不会耦合在一起了，可以独立调整。
    g = IsingGraph(xL, yL, zL, 
        Continuous(), 
        wg5, 
        LatticeConstants(1.0, 1.0, 1.0),
        Ising(b = StateLike(UniformArray,0), localpotential = StateLike(UniformArray,0)) + 
            CoulombHamiltonian(scaling = Scale, screening = Screening, recalc = 1000) + 
            Quartic(localpotential = StateLike(UniformArray,0)) + 
            Sextic(localpotential = StateLike(UniformArray,0)), 
        StateSet(-1.5f0, 1.5f0),
        periodic = (:x,:y),
        diag = StateLike(UniformArray)
    )
    ###这样的话，可以在后续直接调整每一个参数。
        g.hamiltonian[1].lp[] = a1
        g.hamiltonian[1].c[] = 1
        g.hamiltonian[5].lp[] = b1
        g.hamiltonian[5].c[] = 1
        g.hamiltonian[6].lp[] = c1
        g.hamiltonian[6].c[] = 1
=#

g.hamiltonian[Quadratic] 