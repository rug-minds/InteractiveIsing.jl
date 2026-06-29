# AI Generated

export OptimizationBackend, AbstractGPUBackend, AutoBackend, CPUBackend, MetalBackend, AMDGPUBackend

abstract type OptimizationBackend end
abstract type AbstractGPUBackend <: OptimizationBackend end

struct AutoBackend <: OptimizationBackend end
struct CPUBackend{State} <: OptimizationBackend
    state::State
end

struct MetalBackend{BackendModule,State} <: AbstractGPUBackend
    backend_module::BackendModule
    state::State
end

struct AMDGPUBackend{BackendModule,State} <: AbstractGPUBackend
    backend_module::BackendModule
    state::State
end

struct OptimizationBuffers{Backend,X,P,G}
    backend::Backend
    x::X
    p::P
    grad::G
end

"""
    CPUBackend()

Construct a CPU backend selector that reads graph state for standalone
derivative calls.
"""
function CPUBackend()
    return CPUBackend(nothing)
end

"""
    MetalBackend()

Construct an uninitialized Metal backend selector.
"""
function MetalBackend()
    return MetalBackend(nothing, nothing)
end

"""
    AMDGPUBackend()

Construct an uninitialized AMDGPU backend selector.
"""
function AMDGPUBackend()
    return AMDGPUBackend(nothing, nothing)
end

"""
    optimization_backend(backend)

Normalize a user-facing backend selector into an optimization backend value.
"""
function optimization_backend(backend::Symbol)
    backend === :auto && return AutoBackend()
    backend === :cpu && return CPUBackend()
    backend === :metal && return MetalBackend()
    backend === :m1 && return MetalBackend()
    backend === :amd && return AMDGPUBackend()
    backend === :amdgpu && return AMDGPUBackend()
    throw(ArgumentError("Unknown optimization backend `$(backend)`. Use :auto, :cpu, :metal/:m1, or :amd/:amdgpu."))
end

"""
    optimization_backend(backend)

Pass through an already-normalized optimization backend value.
"""
function optimization_backend(backend::B) where {B<:OptimizationBackend}
    return backend
end

"""
    selected_backend(backend)

Resolve automatic backend selection to a concrete backend.
"""
function selected_backend(::AutoBackend)
    if Sys.isapple() && Sys.ARCH in (:aarch64, :arm64)
        return MetalBackend()
    end
    return AMDGPUBackend()
end

"""
    selected_backend(backend)

Keep explicit backend selections unchanged.
"""
function selected_backend(backend::B) where {B<:OptimizationBackend}
    return backend
end

"""
    backend_package_id(backend)

Return the Julia package identity required by a Metal backend.
"""
function backend_package_id(::MetalBackend)
    return Base.PkgId(Base.UUID("dde4c033-4e86-420c-a63e-0dd931031962"), "Metal")
end

"""
    backend_package_id(backend)

Return the Julia package identity required by an AMDGPU backend.
"""
function backend_package_id(::AMDGPUBackend)
    return Base.PkgId(Base.UUID("21141c5a-9bdb-4563-92ae-f87d6854732e"), "AMDGPU")
end

"""
    backend_package_name(backend)

Return the dependency display name for a Metal backend.
"""
function backend_package_name(::MetalBackend)
    return "Metal"
end

"""
    backend_package_name(backend)

Return the dependency display name for an AMDGPU backend.
"""
function backend_package_name(::AMDGPUBackend)
    return "AMDGPU"
end

"""
    require_backend_module(backend)

Return `nothing` for the CPU backend because it needs no external package.
"""
function require_backend_module(::CPUBackend)
    return nothing
end

"""
    require_backend_module(backend)

Load the GPU package required by `backend`.
"""
function require_backend_module(backend::B) where {B<:AbstractGPUBackend}
    pkgid = backend_package_id(backend)
    try
        return Base.require(pkgid)
    catch err
        name = backend_package_name(backend)
        throw(ArgumentError("Optimization backend $(typeof(backend)) requires $(name).jl to be installed and loadable. Original error: $(err)"))
    end
end

"""
    initialized_backend(backend, backend_module, state)

Create an initialized CPU backend for derivative evaluation.
"""
function initialized_backend(::CPUBackend, backend_module, state::S) where {S<:AbstractVector}
    return CPUBackend(state)
end

"""
    initialized_backend(backend, backend_module, state)

Create an initialized Metal backend for derivative evaluation.
"""
function initialized_backend(::MetalBackend, backend_module::Module, state::S) where {S<:AbstractVector}
    return MetalBackend(backend_module, state)
end

"""
    initialized_backend(backend, backend_module, state)

Create an initialized AMDGPU backend for derivative evaluation.
"""
function initialized_backend(::AMDGPUBackend, backend_module::Module, state::S) where {S<:AbstractVector}
    return AMDGPUBackend(backend_module, state)
end

"""
    backend_state(backend, model)

Return the graph state for CPU derivative execution.
"""
function backend_state(backend::CPUBackend, model::M) where {M<:AbstractIsingGraph}
    state = getfield(backend, :state)
    isnothing(state) || return state
    return graphstate(model)
end

"""
    backend_state(backend, model)

Return the active GPU position buffer for GPU derivative execution.
"""
function backend_state(backend::B, model::M) where {B<:AbstractGPUBackend,M<:AbstractIsingGraph}
    return getfield(backend, :state)
end

"""
    backend_array(backend, input)

Return `input` unchanged for CPU execution.
"""
function backend_array(::CPUBackend, input::A) where {A<:AbstractArray}
    return input
end

"""
    backend_array(backend, input)

Move an array-like object to a Metal array.
"""
function backend_array(backend::MetalBackend, input::A) where {A<:AbstractArray}
    return getproperty(getfield(backend, :backend_module), :MtlArray)(input)
end

"""
    backend_array(backend, input)

Move an array-like object to an AMDGPU ROCArray.
"""
function backend_array(backend::AMDGPUBackend, input::A) where {A<:AbstractArray}
    return getproperty(getfield(backend, :backend_module), :ROCArray)(input)
end

"""
    backend_vector(backend, input)

Return a vector representation of `input` on `backend`.
"""
function backend_vector(backend::B, input::A) where {B<:OptimizationBackend,A<:AbstractArray}
    return backend_array(backend, collect(input))
end

"""
    backend_matrix(backend, input)

Return a matrix representation of `input` on `backend`.
"""
function backend_matrix(backend::B, input::A) where {B<:OptimizationBackend,A<:AbstractMatrix}
    return backend_array(backend, Matrix(input))
end

"""
    backend_scalar(value, ::Type{T})

Convert a scalar or zero-dimensional parameter container to scalar type `T`.
"""
function backend_scalar(value, ::Type{T}) where {T<:Real}
    return value isa AbstractArray ? T(value[]) : T(value)
end

"""
    init_optimization_buffers(backend, x0)

Allocate position, momentum, and derivative buffers on the selected backend.
"""
function init_optimization_buffers(backend::B, x0::AbstractVector{T}) where {B<:OptimizationBackend,T<:AbstractFloat}
    selected = selected_backend(backend)
    backend_module = require_backend_module(selected)
    bootstrap = initialized_backend(selected, backend_module, x0)
    x = backend_array(bootstrap, copy(x0))
    active = initialized_backend(selected, backend_module, x)
    p = backend_array(active, zeros(T, length(x0)))
    grad = backend_array(active, similar(x0))
    return OptimizationBuffers(active, x, p, grad)
end

"""
    copy_optimization_position!(dest, buffers)

Copy the current backend position buffer into a CPU vector.
"""
function copy_optimization_position!(dest::AbstractVector{T}, buffers::OptimizationBuffers) where {T<:AbstractFloat}
    copyto!(dest, Array(getfield(buffers, :x)))
    return dest
end
