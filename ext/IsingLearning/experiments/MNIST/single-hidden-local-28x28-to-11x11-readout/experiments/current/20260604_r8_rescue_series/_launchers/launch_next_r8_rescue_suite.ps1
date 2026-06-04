$ErrorActionPreference = "Stop"

$repo = "C:\Users\fenje\dev\InteractiveIsing.jl"
$julia = "C:\Users\fenje\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe"
$script = Join-Path $repo "ext\IsingLearning\experiments\MNIST\single-hidden-local-28x28-to-11x11-readout\mnist_local_manager_grid.jl"
$series = Join-Path $repo "ext\IsingLearning\experiments\MNIST\single-hidden-local-28x28-to-11x11-readout\experiments\current\20260604_r8_rescue_series"
$logdir = Join-Path $series "_launchers\logs"

New-Item -ItemType Directory -Force -Path $logdir | Out-Null
Set-Location $repo

function Set-CommonEnv {
    param(
        [string]$Attempt,
        [string]$Name,
        [int]$Epochs,
        [int]$FreeSweeps,
        [int]$NudgeSweeps,
        [string]$Beta,
        [string]$Lr,
        [string]$LrBias,
        [int]$Seed
    )

    $outdir = Join-Path $series $Attempt
    New-Item -ItemType Directory -Force -Path $outdir | Out-Null

    $env:ISING_MNIST_PM_NAME = $Name
    $env:ISING_MNIST_PM_OUTDIR = $outdir
    $env:ISING_MNIST_PM_RADII = "8"
    $env:ISING_MNIST_PM_RADIUS = "8"
    $env:ISING_MNIST_PM_WORKERS = "32"
    $env:ISING_MNIST_PM_EPOCHS = "$Epochs"
    $env:ISING_MNIST_PM_BATCHSIZE = "128"
    $env:ISING_MNIST_PM_JOB_CHUNK_SIZE = "4"
    $env:ISING_MNIST_PM_TRAIN_PER_CLASS = "100"
    $env:ISING_MNIST_PM_TEST_PER_CLASS = "20"
    $env:ISING_MNIST_PM_FREE_SWEEPS = "$FreeSweeps"
    $env:ISING_MNIST_PM_NUDGE_SWEEPS = "$NudgeSweeps"
    $env:ISING_MNIST_PM_FREE_READS = "3"
    $env:ISING_MNIST_PM_NUDGE_READS = "3"
    $env:ISING_MNIST_PM_BETA = $Beta
    $env:ISING_MNIST_PM_LR_W0 = $Lr
    $env:ISING_MNIST_PM_LR_W12 = $Lr
    $env:ISING_MNIST_PM_LR_W2O = $Lr
    $env:ISING_MNIST_PM_LR_B = $LrBias
    $env:ISING_MNIST_PM_TRAIN_OUTPUT_BIAS = "false"
    $env:ISING_MNIST_PM_TRAIN_INTERNAL = "false"
    $env:ISING_MNIST_PM_DYNAMICS = "metropolis"
    $env:ISING_MNIST_PM_SEED = "$Seed"
    $env:ISING_MNIST_PM_PROGRESS = "true"
    $env:ISING_MNIST_PM_PROGRESS_BAR = "false"
    $env:ISING_MNIST_PM_PROGRESS_EVERY = "5"

    return $outdir
}

function Run-Attempt {
    param(
        [string]$Attempt,
        [string]$Name,
        [int]$Epochs,
        [int]$FreeSweeps,
        [int]$NudgeSweeps,
        [string]$Beta,
        [string]$Lr,
        [string]$LrBias,
        [int]$Seed
    )

    $outdir = Set-CommonEnv -Attempt $Attempt -Name $Name -Epochs $Epochs -FreeSweeps $FreeSweeps -NudgeSweeps $NudgeSweeps -Beta $Beta -Lr $Lr -LrBias $LrBias -Seed $Seed
    $stdout = Join-Path $logdir "$Attempt`_stdout.log"
    $stderr = Join-Path $logdir "$Attempt`_stderr.log"

    "[$(Get-Date -Format s)] START $Attempt outdir=$outdir" | Tee-Object -FilePath (Join-Path $logdir "suite_status.log") -Append
    & $julia -t 32 --project=ext/IsingLearning $script 1>$stdout 2>$stderr
    $code = $LASTEXITCODE
    "[$(Get-Date -Format s)] END $Attempt exit=$code" | Tee-Object -FilePath (Join-Path $logdir "suite_status.log") -Append
    if ($code -ne 0) {
        throw "$Attempt failed with exit code $code; see $stderr"
    }
}

Run-Attempt `
    -Attempt "attempt_04_metropolis_s25_beta2p5_lr000025_b128_chunk4_seed2468_e120" `
    -Name "r8_rescue_metropolis_s25_beta2p5_lr000025_b128_chunk4_seed2468_e120" `
    -Epochs 120 -FreeSweeps 25 -NudgeSweeps 25 -Beta "2.5" -Lr "0.000025" -LrBias "0.0000025" -Seed 2468

Run-Attempt `
    -Attempt "attempt_05_metropolis_s25_beta1p5_lr000025_b128_chunk4_seed2468_e120" `
    -Name "r8_rescue_metropolis_s25_beta1p5_lr000025_b128_chunk4_seed2468_e120" `
    -Epochs 120 -FreeSweeps 25 -NudgeSweeps 25 -Beta "1.5" -Lr "0.000025" -LrBias "0.0000025" -Seed 2468

Run-Attempt `
    -Attempt "attempt_06_metropolis_s25_beta2p5_lr00005_b128_chunk4_seed13579_e80" `
    -Name "r8_rescue_metropolis_s25_beta2p5_lr00005_b128_chunk4_seed13579_e80" `
    -Epochs 80 -FreeSweeps 25 -NudgeSweeps 25 -Beta "2.5" -Lr "0.00005" -LrBias "0.000005" -Seed 13579

Run-Attempt `
    -Attempt "attempt_07_metropolis_s25_beta2p5_lr00005_b128_chunk4_seed31415_e80" `
    -Name "r8_rescue_metropolis_s25_beta2p5_lr00005_b128_chunk4_seed31415_e80" `
    -Epochs 80 -FreeSweeps 25 -NudgeSweeps 25 -Beta "2.5" -Lr "0.00005" -LrBias "0.000005" -Seed 31415

Run-Attempt `
    -Attempt "attempt_08_metropolis_s35_beta2p5_lr00005_b128_chunk4_seed2468_e80" `
    -Name "r8_rescue_metropolis_s35_beta2p5_lr00005_b128_chunk4_seed2468_e80" `
    -Epochs 80 -FreeSweeps 35 -NudgeSweeps 35 -Beta "2.5" -Lr "0.00005" -LrBias "0.000005" -Seed 2468

"[$(Get-Date -Format s)] SUITE COMPLETE" | Tee-Object -FilePath (Join-Path $logdir "suite_status.log") -Append
