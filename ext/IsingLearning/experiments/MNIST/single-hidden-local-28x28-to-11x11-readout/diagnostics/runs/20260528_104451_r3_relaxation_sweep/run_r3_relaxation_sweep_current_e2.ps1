$ErrorActionPreference = "Stop"

$repo = "C:\Users\fenje\dev\InteractiveIsing.jl"
$arch = Join-Path $repo "ext\IsingLearning\experiments\MNIST\single-hidden-local-28x28-to-11x11-readout"
$current = Join-Path $arch "experiments\current"
$checkpoint = Join-Path $arch "diagnostics\runs\r7_scheme_grid\mean_lr004_b32_traininternal_e30\r7\best_model.bin"

$configs = @(
    @{ sweeps = 25; name = "20260528_110043_mnist_single_hidden_local_r3_s25_e2_32w" },
    @{ sweeps = 50; name = "20260528_110043_mnist_single_hidden_local_r3_s50_e2_32w" },
    @{ sweeps = 100; name = "20260528_110043_mnist_single_hidden_local_r3_s100_e2_32w" }
)

$juliaCode = @'
include(raw"C:\Users\fenje\dev\InteractiveIsing.jl\ext\IsingLearning\experiments\MNIST\single-hidden-local-28x28-to-11x11-readout\mnist_local_manager_grid.jl")

Base.@kwdef struct PaperMNISTManagerConfig
    name::String = "r7_manager"
    workers::Int = 32
    epochs::Int = 10
    batchsize::Int = 64
    train_per_class::Int = 100
    test_per_class::Int = 20
    hidden1_side::Int = 28
    hidden2_side::Int = 11
    output_replicas::Int = 4
    local_radius::Int = 7
    internal_radius::Int = 1
    output_internal_radius::Int = 1
    free_reads::Int = 3
    nudge_reads::Int = 3
    free_sweeps::Int = 50
    nudge_sweeps::Int = 50
    β::PMNIST_FT = 5.0f0
    target_on::PMNIST_FT = 1.0f0
    target_off::PMNIST_FT = -1.0f0
    lr_w0::PMNIST_FT = 0.003f0
    lr_w12::PMNIST_FT = 0.003f0
    lr_w2o::PMNIST_FT = 0.003f0
    lr_w11::PMNIST_FT = 0.001f0
    lr_w22::PMNIST_FT = 0.001f0
    lr_woo::PMNIST_FT = 0.001f0
    lr_b::PMNIST_FT = 0.0003f0
    gain_w0::PMNIST_FT = 0.5f0
    gain_w12::PMNIST_FT = 0.25f0
    gain_w2o::PMNIST_FT = 0.25f0
    gain_w11::PMNIST_FT = 0.0f0
    gain_w22::PMNIST_FT = 0.0f0
    gain_woo::PMNIST_FT = 0.0f0
    internal_scale::PMNIST_FT = 0.0f0
    output_internal_scale::PMNIST_FT = 0.0f0
    train_internal::Bool = false
    weight_clip::PMNIST_FT = 1.0f0
    bias_clip::PMNIST_FT = 1.0f0
    applied_bias_clip::PMNIST_FT = 4.0f0
    hot_temp::PMNIST_FT = 5.0f0
    cold_temp::PMNIST_FT = 0.01f0
    reverse_temp::PMNIST_FT = 1.0f0
    gradient_normalization::Symbol = :sum
    seed::Int = 2468
    outdir::String = ""
end

function resume_model!(model::M, path::P) where {M<:LocalMNISTModel,P<:AbstractString}
    isempty(path) && return model
    isfile(path) || throw(ArgumentError("resume checkpoint does not exist: `$path`"))
    saved = open(path, "r") do io
        deserialize(io)
    end

    graph_weights = SparseArrays.nonzeros(II.adj(model.graph))
    if hasproperty(saved, :w) && hasproperty(saved, :b)
        graph_weights .= saved.w
        base_magfield(model.graph).b .= saved.b
        return model
    end

    old_fields = (:weights_0, :weights_12, :weights_2o, :bias_1, :bias_2, :bias_o)
    all(field -> hasproperty(saved, field), old_fields) ||
        throw(ArgumentError("resume checkpoint must contain either `w`/`b` or old matrix weights and biases"))

    function install_bipartite!(group, weights)
        @inbounds for idx in eachindex(group.forward)
            value = -PMNIST_FT(weights[group.srcpos[idx], group.dstpos[idx]])
            graph_weights[group.forward[idx]] = value
            graph_weights[group.reverse[idx]] = value
        end
        return nothing
    end

    install_bipartite!(model.edge_groups.input_hidden, saved.weights_0)
    install_bipartite!(model.edge_groups.hidden_hidden, saved.weights_12)
    install_bipartite!(model.edge_groups.hidden_output, saved.weights_2o)

    b = base_magfield(model.graph).b
    fill!(b, 0f0)
    b[model.hidden1_idxs] .= .-PMNIST_FT.(saved.bias_1)
    b[model.hidden2_idxs] .= .-PMNIST_FT.(saved.bias_2)
    b[model.output_idxs] .= .-PMNIST_FT.(saved.bias_o)
    return model
end

config = LocalMNISTManagerConfig(; name = ENV["ISING_MNIST_PM_NAME"], local_radius = 3, outdir = ENV["ISING_MNIST_PM_OUTDIR"])
run_config!(config)
'@

Push-Location $repo
try {
    foreach ($cfg in $configs) {
        $outdir = Join-Path $current $cfg.name
        New-Item -ItemType Directory -Force -Path $outdir | Out-Null

        $env:ISING_MNIST_PM_NAME = "r3_sweeps_$($cfg.sweeps)"
        $env:ISING_MNIST_PM_WORKERS = "32"
        $env:ISING_MNIST_PM_EPOCHS = "2"
        $env:ISING_MNIST_PM_BATCHSIZE = "32"
        $env:ISING_MNIST_PM_RADIUS = "3"
        $env:ISING_MNIST_PM_FREE_SWEEPS = "$($cfg.sweeps)"
        $env:ISING_MNIST_PM_NUDGE_SWEEPS = "$($cfg.sweeps)"
        $env:ISING_MNIST_PM_FREE_READS = "3"
        $env:ISING_MNIST_PM_NUDGE_READS = "3"
        $env:ISING_MNIST_PM_OPTIMIZER = "adam"
        $env:ISING_MNIST_PM_LR_W0 = "0.004"
        $env:ISING_MNIST_PM_LR_W12 = "0.004"
        $env:ISING_MNIST_PM_LR_W2O = "0.004"
        $env:ISING_MNIST_PM_LR_B = "0.0004"
        $env:ISING_MNIST_PM_GRADIENT_NORMALIZATION = "mean"
        $env:ISING_MNIST_PM_RESUME_CHECKPOINT = $checkpoint
        $env:ISING_MNIST_PM_OUTDIR = $outdir
        $env:ISING_MNIST_PM_PROGRESS = "true"
        $env:ISING_MNIST_PM_PROGRESS_EVERY = "10"

        $stdout = Join-Path $outdir "stdout.log"
        $stderr = Join-Path $outdir "stderr.log"
        $started = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
        "[$started] starting r3 sweeps=$($cfg.sweeps) outdir=$outdir" | Set-Content -Encoding UTF8 -Path $stdout

        & julia -t 32 --project=ext\IsingLearning -e $juliaCode 1>> $stdout 2> $stderr
        $exit = $LASTEXITCODE
        $finished = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
        "[$finished] finished r3 sweeps=$($cfg.sweeps) exit=$exit" | Add-Content -Encoding UTF8 -Path $stdout
        if ($exit -ne 0) {
            throw "r3 sweeps=$($cfg.sweeps) failed with exit code $exit"
        }
    }
}
finally {
    Pop-Location
}
