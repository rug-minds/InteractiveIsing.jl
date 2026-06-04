$ErrorActionPreference = "Stop"

$repo = "C:\Users\fenje\dev\InteractiveIsing.jl"
$julia = "C:\Users\fenje\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe"
$series = Join-Path $repo "ext\IsingLearning\experiments\MNIST\single-hidden-local-28x28-to-11x11-readout\experiments\current\20260604_r8_rescue_series"
$attempt = Join-Path $series "attempt_03_metropolis_s25_beta2p5_lr00005_b128_chunk4_seed2468_e80"

New-Item -ItemType Directory -Force -Path $attempt | Out-Null

$env:ISING_MNIST_PM_NAME = "r8_rescue_metropolis_s25_beta2p5_lr00005_b128_chunk4_seed2468_e80"
$env:ISING_MNIST_PM_OUTDIR = $attempt
$env:ISING_MNIST_PM_RADII = "8"
$env:ISING_MNIST_PM_RADIUS = "8"
$env:ISING_MNIST_PM_WORKERS = "32"
$env:ISING_MNIST_PM_EPOCHS = "80"
$env:ISING_MNIST_PM_BATCHSIZE = "128"
$env:ISING_MNIST_PM_JOB_CHUNK_SIZE = "4"
$env:ISING_MNIST_PM_TRAIN_PER_CLASS = "100"
$env:ISING_MNIST_PM_TEST_PER_CLASS = "20"
$env:ISING_MNIST_PM_FREE_SWEEPS = "25"
$env:ISING_MNIST_PM_NUDGE_SWEEPS = "25"
$env:ISING_MNIST_PM_FREE_READS = "3"
$env:ISING_MNIST_PM_NUDGE_READS = "3"
$env:ISING_MNIST_PM_BETA = "2.5"
$env:ISING_MNIST_PM_LR_W0 = "0.00005"
$env:ISING_MNIST_PM_LR_W12 = "0.00005"
$env:ISING_MNIST_PM_LR_W2O = "0.00005"
$env:ISING_MNIST_PM_LR_B = "0.000005"
$env:ISING_MNIST_PM_TRAIN_OUTPUT_BIAS = "false"
$env:ISING_MNIST_PM_TRAIN_INTERNAL = "false"
$env:ISING_MNIST_PM_DYNAMICS = "metropolis"
$env:ISING_MNIST_PM_SEED = "2468"
$env:ISING_MNIST_PM_PROGRESS = "true"
$env:ISING_MNIST_PM_PROGRESS_BAR = "false"
$env:ISING_MNIST_PM_PROGRESS_EVERY = "5"

Set-Location $repo
& $julia -t 32 --project=ext/IsingLearning "ext/IsingLearning/experiments/MNIST/single-hidden-local-28x28-to-11x11-readout/mnist_local_manager_grid.jl"
