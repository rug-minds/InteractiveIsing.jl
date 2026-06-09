$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..\..\..")
$Project = Join-Path $RepoRoot "ext\IsingLearning"
$Script = Join-Path $ScriptDir "mnist_lcl_6x6_window_adam.jl"
$OutDir = Join-Path $ScriptDir "experiments\current\smoke_lcl_6x6_stride1"

$env:ISING_MNIST_IF_WORKERS = "2"
$env:ISING_MNIST_IF_EPOCHS = "1"
$env:ISING_MNIST_IF_BATCHSIZE = "4"
$env:ISING_MNIST_IF_TRAIN_PER_CLASS = "2"
$env:ISING_MNIST_IF_TEST_PER_CLASS = "1"
$env:ISING_MNIST_IF_TRAIN_EVAL_PER_CLASS = "1"
$env:ISING_MNIST_IF_EVAL_EVERY = "1"
$env:ISING_MNIST_IF_SWEEPS = "1"
$env:ISING_MNIST_IF_BETA = "0.1"
$env:ISING_MNIST_LCL_TANGENT_NUDGE = "true"
$env:ISING_MNIST_IF_OUTDIR = $OutDir

julia -t 2 "--project=$Project" $Script
