$ErrorActionPreference = "Stop"

$repo = "C:\Users\fenje\dev\InteractiveIsing.jl"
$julia = "C:\Users\fenje\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe"
$script = "ext/IsingLearning/experiments/MNIST/single-hidden-local-28x28-to-11x11-readout/mnist_local_manager_grid.jl"
$series = Join-Path $repo "ext\IsingLearning\experiments\MNIST\single-hidden-local-28x28-to-11x11-readout\experiments\current\20260604_r8_rescue_series"
$logdir = Join-Path $series "_launchers\logs"
$status = Join-Path $logdir "retry_suite_02_status.log"

New-Item -ItemType Directory -Force -Path $logdir | Out-Null
Set-Location $repo

function Configure-R8Attempt {
    param(
        [string]$Attempt,
        [string]$Name,
        [int]$Epochs,
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
    $env:ISING_MNIST_PM_FREE_SWEEPS = "25"
    $env:ISING_MNIST_PM_NUDGE_SWEEPS = "25"
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

function Run-R8Attempt {
    param(
        [string]$Attempt,
        [string]$Name,
        [int]$Epochs,
        [string]$Beta,
        [string]$Lr,
        [string]$LrBias,
        [int]$Seed
    )

    $outdir = Configure-R8Attempt -Attempt $Attempt -Name $Name -Epochs $Epochs -Beta $Beta -Lr $Lr -LrBias $LrBias -Seed $Seed
    $stdout = Join-Path $logdir "$Attempt`_stdout.log"
    $stderr = Join-Path $logdir "$Attempt`_stderr.log"

    Add-Content -Path $status -Value "[$(Get-Date -Format s)] START $Attempt outdir=$outdir"
    $proc = Start-Process -FilePath $julia -ArgumentList @("-t", "32", "--project=ext/IsingLearning", $script) -WorkingDirectory $repo -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
    Add-Content -Path $status -Value "[$(Get-Date -Format s)] PID $Attempt $($proc.Id)"
    Wait-Process -Id $proc.Id
    Add-Content -Path $status -Value "[$(Get-Date -Format s)] END $Attempt"
}

Run-R8Attempt `
    -Attempt "attempt_09_metropolis_s25_beta1p5_lr000025_b128_chunk4_seed2468_e200" `
    -Name "r8_retry_metropolis_s25_beta1p5_lr000025_b128_chunk4_seed2468_e200" `
    -Epochs 200 -Beta "1.5" -Lr "0.000025" -LrBias "0.0000025" -Seed 2468

Run-R8Attempt `
    -Attempt "attempt_10_metropolis_s25_beta1p0_lr000025_b128_chunk4_seed2468_e200" `
    -Name "r8_retry_metropolis_s25_beta1p0_lr000025_b128_chunk4_seed2468_e200" `
    -Epochs 200 -Beta "1.0" -Lr "0.000025" -LrBias "0.0000025" -Seed 2468

Run-R8Attempt `
    -Attempt "attempt_11_metropolis_s25_beta1p5_lr00001_b128_chunk4_seed2468_e240" `
    -Name "r8_retry_metropolis_s25_beta1p5_lr00001_b128_chunk4_seed2468_e240" `
    -Epochs 240 -Beta "1.5" -Lr "0.00001" -LrBias "0.000001" -Seed 2468

Run-R8Attempt `
    -Attempt "attempt_12_metropolis_s25_beta1p0_lr00001_b128_chunk4_seed2468_e240" `
    -Name "r8_retry_metropolis_s25_beta1p0_lr00001_b128_chunk4_seed2468_e240" `
    -Epochs 240 -Beta "1.0" -Lr "0.00001" -LrBias "0.000001" -Seed 2468

Add-Content -Path $status -Value "[$(Get-Date -Format s)] RETRY SUITE 02 COMPLETE"
