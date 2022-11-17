qmlfile = joinpath( @__DIR__, "qml", "Main.qml")

# For plotting
const img =  Ref(zeros(RGB{Float64},1,1))

function showlatest(buffer::Array{UInt32, 1}, width32::Int32, height32::Int32)
    buffer = reshape(buffer, size(img[]))
    buffer = reinterpret(ARGB32, buffer)
    buffer .= transpose(img[])
    return
end

# Simulation struct
export IsingSim
mutable struct IsingSim
    # Graph
    const g::IsingGraph
    # Property map for qml
    const pmap::JuliaPropertyMap
    
    # length/width of graph
    const gSize::Observable{Int32}

    # Temperature Observable
    const TIs::Observable{Float32}

    # For drawing to simulation
    const brush::Observable{Float32} 
    const brushR::Observable{Int32} 
    const circ::Observable 

    # Magnetization
    const M::Observable{Float32} 
    const M_array::Ref{Vector{Real}}

    const analysisRunning::Observable{Bool} 
    
    # For tracking updates
    updates::Int
    # Updates per frame average
    const upf::Observable{Int} 

    # Image of graph
    const img::Base.RefValue{Matrix{RGB{Float64}}}
    const imgSize::Observable

    # Thread Locking
    const updatingUpf::Ref{Bool}
    const updatingMag::Ref{Bool} 
    const updatingImg::Ref{Bool} 

    # For Branching Simulation
    const shouldRun::Observable{Bool} 
    isRunning::Bool

    # Hamiltonian factor
    efac::Function

    function IsingSim(;
            continuous = false,
            graphSize = 512,
            weighted = true,
            weightFunc = defaultIsingWF,
            initTemp = 1.,
            start = false
        );

        type = continuous ? Float32 : Int8

        g = IsingGraph(
            graphSize,
            continuous = continuous, 
            weighted = weighted,
            weightFunc = weighted ? weightFunc : defaultIsingWF
        )
        
        initImg = gToImg(g)
        initbrushR= round(graphSize/10)

        sim = new(
            g,
            JuliaPropertyMap(),
            Observable(Int32(graphSize)),
            Observable(Float32(initTemp)),
            Observable(Float32(0.)),
            Observable( Int32(initbrushR) ),
            Observable(getOrdCirc(Int32(initbrushR))),
            Observable(Float32(0.0)),
            zeros(Real,60),
            Observable(false),
            0,
            Observable(0),
            img,
            Observable(size(initImg)),
            Ref(false),
            Ref(false),
            Ref(false),
            Observable(true),
            true
        )

        # Initialize image
        sim.img[] = initImg

        # Set hamiltonian factor function
        sim.efac = () -> 1

        # Initializing propertymap
        sim.pmap["imgSize"] = sim.imgSize
        sim.pmap["shouldRun"] = sim.shouldRun
        sim.pmap["TIs"] = sim.TIs
        sim.pmap["brush"] = sim.brush
        sim.pmap["brushR"] = sim.brushR
        sim.pmap["circ"] = sim.circ 
        sim.pmap["M"] = sim.M
        sim.pmap["analysisRunning"] = sim.analysisRunning
        sim.pmap["upf"] = sim.upf
        sim.pmap["gSize"] = sim.gSize


        if start
            s()
        end
        return sim
    end

    
end

"""
Start the simulation and interface
Will add a non-interface mode soon.
"""
function (sim::IsingSim)(start = true; async = true)
    if start
        if Threads.nthreads() < 4
            error("Please enable multithreading to use the interface. For documentation, see github page")
        else
            startSim(sim; async)
        end
    end
    return sim.g;
end



#= 
Persistent functions of the simulation 
=#

"""
Main loop for for MCMC
When a new getE function needs to be defined, this loop can be branched to a new loop with a new getE func
Depends on two variables, isRunning and shouldRun to check wether current branch is up to date or not
When another thread needs to invalidate branch, it sets shouldRun to false
Then it waits until isRunning is set to false after which shouldRun can be activated again.
Then, this function itself makes a new branch where getE is defined again.
export updateGraph
"""
function updateGraph(sim::IsingSim)
    g = sim.g
    TIs = sim.TIs
    htype = g.htype
    
    rng = MersenneTwister()
    g_iterator = ising_it(g,htype)
    shouldRun = sim.shouldRun
    
    # Defining argumentless functions here seems faster.
    # Offset large function into files for clearity
    # @includetextfile MonteCarlo updateMonteCarloIsingD
    function updateMonteCarloIsingD()
        beta = 1/(TIs[])
        
        idx = rand(rng, g_iterator)
        
        Estate = @inbounds g.state[idx]*getEFactor(g, idx, g.htype)
        
        if (Estate >= 0 || rand(rng) < exp(2*beta*Estate))
            @inbounds g.state[idx] *= -1
        end
        
    end

    # @includetextfile MonteCarlo updateMonteCarloIsingC


    # isingUpdate = typeof(g) == IsingGraph{Int8} ? 
    #         updateMonteCarloIsingD : updateMonteCarloIsingC

    sim.isRunning = true

    while shouldRun[]
        updateMonteCarloIsingD()
        sim.updates += 1
        
        GC.safepoint()
    end

    sim.isRunning = false
    while !shouldRun[]
        yield()
    end
    updateGraph(sim)

end
export updateGraph


function reInitSim(sim)
    g = sim.g
    g.state = typeof(g) == IsingGraph{Int8} ? initRandomState(g.size) : initRandomCState(g.size)
    editHType!(g, :MagField => false, :Defects => false, :Clamp => false)
    g.d.defectBools = [false for x in 1:g.size]
    g.d.defectList = []
    g.d.aliveList = [1:g.size;]
    g.d.mlist = zeros(Float32, g.size)

    sim.M[] = 0
    sim.updates = 0

    branchSim(sim)
end

# Timed Functions 
# Updating image of graph
export updateImg
function updateImg(sim)
    sim.img[] = gToImg(sim.g)
    return
end

# Track number of updates per frame
let avgWindow = 60, updateWindow = zeros(Int64,avgWindow), frames = 0
    global function updatesPerFrame(sim::IsingSim)
        updateWindow = insertShift(updateWindow,sim.updates)
        if frames > avgWindow
            sim.upf[] = round(sum(updateWindow)/avgWindow)
            frames = 0
        end
        frames += 1
        sim.updates = 0
    end
end

# Averages M_array over an amount of steps
# Updates magnetization (which is thus the averaged value)
let avg_window = 60, frames = 0
    global function magnetization(sim::IsingSim)
        avg_window = 60 # Averaging window = Sec * FPS, becomes max length of vector
        sim.M_array[] = insertShift(sim.M_array[], sum(sim.g.state))
        if frames > avg_window
            sim.M[] = sum(sim.M_array[])/avg_window 
            frames = 0
        end 
        frames += 1 
    end
end

# Pauses sim and waits until paused
export pauseSim
function pauseSim(sim)
    sim.shouldRun[] = false

    while sim.isRunning[]
        yield()
    end

    return true
end

export unpauseSim
function unpauseSim(sim)
    sim.isRunning && return

    sim.shouldRun[] = true

    while !sim.isRunning
        yield()
    end

    return true
end

# """ QML FUNCTIONS """
function annealing(sim, Ti, Tf, initWait = 30, stepWait = 5; Tstep = .5, T_it = Ti:Tstep:Tf, reInit = true, saveImg = true)
    # Reinitialize
    reInit && initIsing()

    # Set temp and initial wait
    TIs[] = Ti
    sleep(initWait)
    
    for temp in T_it
        TIs[] = temp
        sleep(stepWait)
        if saveImg
            save(File{format"PNG"}("Images/Annealing/Ising T$temp.PNG"), img[])
        end
    end
end

#= 
For QML canvas to show image 
=#

export setRenderLoop
function setRenderLoop()
    ENV["QSG_RENDER_LOOP"] = "basic"
end

# Defines all functions that can be run from QML interface
export qmlFunctions
function qmlFunctions(sim::IsingSim)
    g = sim.g
    circ = sim.circ
    brush = sim.brush
    TIs = sim.TIs
    M_array = sim.M_array
    M = sim.M
    brushR = sim.brushR

    # Locks
    updatingImg = sim.updatingImg
    updatingUpf = sim.updatingUpf
    updatingMag = sim.updatingMag

    analysisRunning = sim.analysisRunning

    @qmlfunction println

    # All functions that are run from the QML Timer
    function timedFunctions()
        spawnOne(updateImg, updatingImg, "UpdateImg", sim)
        spawnOne(updatesPerFrame, updatingUpf, "", sim)
        spawnOne(magnetization, updatingMag, "", sim)
    end
    @qmlfunction timedFunctions


    # Add percentage of defects to lattice
    addRandomDefectsQML(pDefects) = addRandomDefects!(sim, g,pDefects)
    @qmlfunction addRandomDefectsQML

    # Initialize isinggraph and display
    function initIsing()
        reInitSim(sim) 
    end
    @qmlfunction initIsing

    # Draw circle to state
    circleToStateQML(i,j,clamp=false) = errormonitor(Threads.@spawn circleToState(sim, g,circ[],i,j,brush[]; clamp, imgsize = size(img[])[1]))
    @qmlfunction circleToStateQML

    # Sweep temperatures and record magnetization and correlation lengths
    # Make an interface for this
    function tempSweepQML(TI = TIs[], TF = 13, TStep = 0.5, dpoints = 12, dpointwait = 5, stepwait = 0, equiwait = 0 , saveImg = true)
        if !g.d.defects
            corrF = sampleCorrPeriodic
        else
            corrF = sampleCorrPeriodicDefects
        end
        # Catching error doesn't work like this, why?
        Threads.@spawn errormonitor(tempSweep(g,TIs,M_array; TI,TF,TStep, dpoints , dpointwait, stepwait, equiwait, saveImg, img=img, analysisRunning=analysisRunning, corrF))
    end
    @qmlfunction tempSweepQML


    # Save a new circle with size brushR[]
    function newCirc()
        circ[] = getOrdCirc(brushR[])
    end
    @qmlfunction newCirc

    saveGImgQML() = saveGImg(g)
    @qmlfunction saveGImgQML
end

#= 
IsingSim Functions
=#

export runSim
# Spawn graph update thread and load qml interface
function runSim(sim; async)
    # showlatest_cfunction = showlatesteval(sim)
    Threads.@spawn errormonitor(updateGraph(sim))
    loadqml( qmlfile, obs = sim.pmap, showlatest = showlatest_cfunction)
    if async
        exec_async()
    else
        exec()
    end
end

export startSim
# Set the render loop,
# Define qml functions
# Start graph update and qml interface
function startSim(sim; async)
    setRenderLoop()
    qmlFunctions(sim)
    runSim(sim; async)
end

# """ REPL FUNCTIONS FOR DEBUGGING """

# # # Draw circle to state
# circleToStateQML(i,j,clamp=false) = Threads.@spawn circleToState(g,circ[],i,j,brush[]; clamp, imgsize = size(img[])[1])
circleToStateREPL(i,j, clamp = false) = circleToState(g,circ[],i,j,brush[]; clamp, imgsize = size(img[])[1])

function tempSweepQMLRepl(TI = TIs[], TF = 13, TStep = 0.5, dpoints = 12, dpointwait = 5, stepwait = 0, equiwait = 0 , saveImg = true); analysisRunning[] = true; tempSweep(g,TIs,M_array; TI,TF,TStep, dpoints , dpointwait, stepwait, equiwait, saveImg, img=img, analysisRunning=analysisRunning, savelast = true) end
