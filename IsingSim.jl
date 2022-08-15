# __precompile__()

module IsingSim

qmlfile = joinpath(@__DIR__, "qml", "Main.qml")


push!(LOAD_PATH, pwd())
push!(LOAD_PATH, pwd()*"/Interaction")
push!(LOAD_PATH, pwd()*"/Learning")

using QML
export QML

using LinearAlgebra, Distributions, Random, GLMakie, FileIO,  Observables, ColorSchemes, Images, DataFrames, CSV, BenchmarkTools, CxxWrap
import Plots as pl

using SquareAdj, WeightFuncs, IsingGraphs, Interaction, Analysis, IsingMetropolis, GPlotting
using IsingLearning

export GPlotting
export IsingGraphs
export Interaction
export IsingMetropolis
export WeightFuncs
export Analysis
export IsingLearning

# For plotting
const img =  Ref(zeros(RGB{Float64},1,1))

function showlatest(buffer::Array{UInt32, 1}, width32::Int32, height32::Int32)
    buffer = reshape(buffer, size(img[]))
    buffer = reinterpret(ARGB32, buffer)
    buffer .= transpose(img[])
    return
end

# Helper to call a julia function
function julia_call(f, argptr::Ptr{Cvoid})
    arglist = CxxRef{QVariantList}(argptr)[]
    result = QVariant(f((value(x) for x in arglist)...))
    return result.cpp_object
end

function __init__()
    global showlatest_cfunction = CxxWrap.@safe_cfunction(showlatest, Cvoid, 
                                               (Array{UInt32,1}, Int32, Int32))
end
export showlatest_cfunction

# Simulation struct
export Sim
mutable struct Sim
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

    function Sim(;
            continuous = false,
            graphSize = 512,
            weighted = true,
            weightFunc = defaultIsingWF,
            initTemp = 1.,
            start = false
        );
        if continuous
            g = CIsingGraph(
                    graphSize, 
                    weighted = weighted,
                    weightFunc = weighted ? weightFunc : defaultIsingWF
                )
        else
            g = IsingGraph(
                graphSize, 
                weighted = weighted,
                weightFunc = weighted ? weightFunc : defaultIsingWF
            )
        end
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
            # Ref(initImg),
            Observable(size(initImg)),
            Ref(false),
            Ref(false),
            Ref(false),
            Observable(true),
            true,
        )
        sim.img[] = initImg
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

function (sim::Sim)(start = true)
    if start
        startSim(sim)
    end
    return sim.g
end



""" Persistent functions of the simulation """

# Main loop for for MCMC
# When a new getE function needs to be defined, this loop can be branched to a new loop with a new getE func
# Depends on two variables, isRunning and shouldRun to check wether current branch is up to date or not
# When another thread needs to invalidate branch, it sets shouldRun to false
# Then it waits until isRunning is set to false after which shouldRun can be activated again.
# Then, this function itself makes a new branch where getE is defined again.
export updateGraph
function updateGraph(sim::Sim)
        g = sim.g
        TIs = sim.TIs
        getE = g.d.hFuncRef[]

        # Defining argumentless functions here seems faster.
        function updateMonteCarloIsingD!()
            T = TIs[]
            @inline function deltE(Estate)
                return -2*Estate
            end
            
            beta = T>0 ? 1/T : Inf
            
            idx = rand(ising_it(g))
            
            Estate = g.state[idx]*getE(g,idx)
            
            if (Estate >= 0 || rand() < exp(-beta*deltE(Estate)))
                @inbounds g.state[idx] *= -1
            end
            
        end

        function updateMonteCarloIsingC!()
            T = TIs[]
            @inline function deltE(efac,newstate,oldstate)
                return efac*(newstate-oldstate)
            end
        
            @inline function sampleCState()
                Float32(2*(rand()-.5))
            end
        
            beta = T>0 ? 1/T : Inf
        
            idx = rand(ising_it(g))
             
            oldstate = g.state[idx]
        
            efactor = getE(g,idx, oldstate)
        
            newstate = sampleCState()
            
            Ediff = deltE(efactor,newstate,oldstate)
            if (Ediff < 0 || rand() < exp(-beta*Ediff))
                @inbounds g.state[idx] = newstate 
            end
            
        end

        if typeof(g) == IsingGraph{Int8}
            isingUpdate = updateMonteCarloIsingD!
        else
            isingUpdate = updateMonteCarloIsingC!
        end

        # Old update
        # updatefunc(g,T) = updateMonteCarloIsing!(g,T;getE)
        sim.isRunning = true
        while sim.shouldRun[]
            isingUpdate()

            # Old update
            # updatefunc(g,TIs[])

            sim.updates += 1
            GC.safepoint()
        end

        sim.isRunning = false
        while !sim.shouldRun[]
            yield()
        end
        updateGraph(sim)
end

function reInitSim(sim)
    g = sim.g
    g.state = typeof(g) == IsingGraph{Int8} ? initRandomState(g.size) : initRandomCState(g.size)
    g.d.defects = false
    g.d.defectBools = [false for x in 1:g.size]
    g.d.defectList = []
    g.d.aliveList = [1:g.size;]
    g.d.mactive = false
    g.d.mlist = zeros(Float32, g.size)
    g.d.hFuncRef = g.d.weighted ? Ref(HFunc) : Ref(HWeightedFunc)

    sim.M[] = 0
    sim.updates = 0

    # This makes the function run if paused before
    # Fix that
    setGHFunc!(sim)
end

"""Timed Functions"""
# Updating image of graph
export updateImg
function updateImg(sim)
    sim.img[] = gToImg(sim.g)
    return
end

# Track number of updates per frame
let avgWindow = 60, updateWindow = zeros(Int64,avgWindow), frames = 0
    global function updatesPerFrame(sim::Sim)
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
    global function magnetization(sim::Sim)
        avg_window = 60 # Averaging window = Sec * FPS, becomes max length of vector
        sim.M_array[] = insertShift(sim.M_array[], sum(sim.g.state))
        if frames > avg_window
            sim.M[] = sum(sim.M_array[])/avg_window 
            frames = 0
        end 
        frames += 1 
    end
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

""" For QML canvas to show image """

export setRenderLoop
function setRenderLoop()
    ENV["QSG_RENDER_LOOP"] = "basic"
end

# Defines all functions that can be run from QML interface
export qmlFunctions
function qmlFunctions(sim::Sim)
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
        spawnOne(updateImg, updatingImg, sim)
        spawnOne(updatesPerFrame, updatingUpf, sim)
        spawnOne(magnetization, updatingMag, sim)
    end
    @qmlfunction timedFunctions


    # Add percentage of defects to lattice
    addRandomDefectsQML(pDefects) = addRandomDefects!(g,pDefects)
    @qmlfunction addRandomDefectsQML

    # Initialize isinggraph and display
    function initIsing()
        reInitSim(sim) 
    end
    @qmlfunction initIsing
    # Draw circle to state
    circleToStateQML(i,j,clamp=false) = Threads.@spawn circleToState(g,circ[],i,j,brush[]; clamp, imgsize = size(img[])[1])
    @qmlfunction circleToStateQML

    # Sweep temperatures and record magnetization and correlation lengths
    # Make an interface for this
    function tempSweepQML(TI = TIs[], TF = 13, TStep = 0.5, dpoints = 12, dpointwait = 5, stepwait = 0, equiwait = 0 , saveImg = true)
        if !g.d.defects
            corrF = sampleCorrPeriodic
        else
            corrF = sampleCorrPeriodicDefects
        end
        Threads.@spawn tempSweep(g,TIs,M_array; TI,TF,TStep, dpoints , dpointwait, stepwait, equiwait, saveImg, img=img, analysisRunning=analysisRunning, corrF)
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

""" Sim Functions"""

export runSim
function runSim(sim)
    # showlatest_cfunction = showlatesteval(sim)
    Threads.@spawn updateGraph(sim)
    loadqml( qmlfile, obs = sim.pmap, showlatest = showlatest_cfunction) 
    exec_async()
end

export startSim
function startSim(sim)
    setRenderLoop()
    qmlFunctions(sim)
    runSim(sim)
end


# """ Helper Functions """

# Insert a value in front of vector and push out last one.
function insertShift(vec::Vector{T}, el::T) where T
    newVec = Vector{T}(undef, length(vec))
    newVec[1:(end-1)] = vec[2:end]
    newVec[end] = el
    return newVec
end

# Spawn a new thread for a function, but only if no thread for that function was already created
# The function is "locked" using a reference to a Boolean value: spawned
function spawnOne(f::Function, spawned::Ref{Bool}, args...)
    # Run function, when thread is finished mark it as not running
    function runOne(func::Function, spawned::Ref{Bool}, args...)
        func(args...)
        spawned[] = false
        GC.safepoint()
    end

    # Mark as running, then spawn thread
    if !spawned[]
        spawned[] = true
        # Threads.@spawn runOne(f,spawned)
        runOne(f,spawned, args...)
    end
end

""" Magnetic field stuff """
function branchSim(sim)
    sim.shouldRun[] = false 
    while sim.isRunning[]
        sleep(.1)
    end
    sim.shouldRun[] = true
end

function setGHFunc!(sim, prt = true)
    g = sim.g
    if !g.d.weighted
        if !g.d.mactive
            g.d.hFuncRef = Ref(HFunc)
            if prt
                println("Set HFunc")
            end
        else
            g.d.hFuncRef = Ref(HMagFunc)
            if prt
                println("Set HMagFunc")
            end
        end
    else
        if !g.d.mactive
            g.d.hFuncRef = Ref(HWeightedFunc)
            if prt
                println("Set HWeightedFunc")
            end
        else
            g.d.hFuncRef = Ref(HWMagFunc)
            if prt
                println("Set HWMagFunc")
            end
        end
    end

    branchSim(sim)
end

export setMIdxs!
function setMIdxs!(sim,idxs,strengths)
    g = sim.g
    if length(idxs) != length(strengths)
        error("Idxs and strengths lengths not the same")
        return      
    end

    g.d.mactive = true
    g.d.mlist[idxs] = strengths
    setGHFunc!(sim, false)
end

# Insert func(;x,y)
export setMFunc!
function setMFunc!(sim,func::Function)
    g = sim.g
    m_matr = Matrix{Float32}(undef,g.N,g.N)
    for y in 1:g.N
        for x in 1:g.N
            m_matr[y,x] = func(;x,y)
        end
    end
    setMIdxs!(sim,[1:g.size;],reshape(transpose(m_matr),g.size))
end

# Removes magnetic field
export remM!
function remM!(sim)
    g = sim.g
    g.d.mactive = false
    setGHFunc!(sim, false)
end

# Set a time dependent magnetic field function f(;x,y,t)
export setMFuncTimed!
function setMFuncTimed!(sim,func::Function, interval = 5, t_step = .2)
    g = sim.g
    for t in 0:t_step:interval
        newfunc(;x,y) = func(;x,y,t)
        setMFunc!(sim,newfunc)
        sleep(t_step)
    end
end


# """ REPL FUNCTIONS FOR DEBUGGING """

# # # Draw circle to state
# circleToStateQML(i,j,clamp=false) = Threads.@spawn circleToState(g,circ[],i,j,brush[]; clamp, imgsize = size(img[])[1])
circleToStateREPL(i,j, clamp = false) = circleToState(g,circ[],i,j,brush[]; clamp, imgsize = size(img[])[1])

function tempSweepQMLRepl(TI = TIs[], TF = 13, TStep = 0.5, dpoints = 12, dpointwait = 5, stepwait = 0, equiwait = 0 , saveImg = true); analysisRunning[] = true; tempSweep(g,TIs,M_array; TI,TF,TStep, dpoints , dpointwait, stepwait, equiwait, saveImg, img=img, analysisRunning=analysisRunning, savelast = true) end

end