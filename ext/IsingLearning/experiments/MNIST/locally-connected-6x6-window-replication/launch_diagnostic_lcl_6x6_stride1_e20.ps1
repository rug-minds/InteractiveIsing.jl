$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..\..\..")
$Project = Join-Path $RepoRoot "ext\IsingLearning"
$Script = Join-Path $ScriptDir "mnist_lcl_6x6_window_adam.jl"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutDir = Join-Path $ScriptDir ("experiments\current\" + $Stamp + "_mnist_lcl_6x6_stride1_diagnostic_e20")

$env:ISING_MNIST_IF_WORKERS = "32"
$env:ISING_MNIST_IF_EPOCHS = "20"
$env:ISING_MNIST_IF_BATCHSIZE = "100"
$env:ISING_MNIST_IF_TRAIN_PER_CLASS = "50"
$env:ISING_MNIST_IF_TEST_PER_CLASS = "20"
$env:ISING_MNIST_IF_TRAIN_EVAL_PER_CLASS = "20"
$env:ISING_MNIST_IF_EVAL_EVERY = "2"
$env:ISING_MNIST_IF_HIDDEN = "529"
$env:ISING_MNIST_IF_OUTPUT_REPLICAS = "1"
$env:ISING_MNIST_IF_LR = "0.0001"
$env:ISING_MNIST_IF_BETA = "0.1"
$env:ISING_MNIST_IF_SWEEPS = "25"
$env:ISING_MNIST_LCL_WINDOW = "6"
$env:ISING_MNIST_LCL_STRIDE = "1"
$env:ISING_MNIST_LCL_TANGENT_NUDGE = "true"
$env:ISING_MNIST_IF_OUTDIR = $OutDir

julia -t 32 "--project=$Project" $Script
