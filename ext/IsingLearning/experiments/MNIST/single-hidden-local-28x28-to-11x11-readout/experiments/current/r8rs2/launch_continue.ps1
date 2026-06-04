$ErrorActionPreference = "Stop"

$repo = "C:\Users\fenje\dev\InteractiveIsing.jl"
$julia = "C:\Users\fenje\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe"
$script = "ext/IsingLearning/experiments/MNIST/single-hidden-local-28x28-to-11x11-readout/mnist_local_manager_grid.jl"
$series = Join-Path $repo "ext\IsingLearning\experiments\MNIST\single-hidden-local-28x28-to-11x11-readout\experiments\current\r8rs2"
$logdir = Join-Path $series "logs"
$status = Join-Path $logdir "status.log"
$a15Checkpoint = Join-Path $repo "ext\IsingLearning\experiments\MNIST\single-hidden-local-28x28-to-11x11-readout\experiments\current\20260604_r8_rescue_series\a15_s50_b10_lr10_e200\r8\latest_checkpoint.bin"

New-Item -ItemType Directory -Force -Path $logdir | Out-Null
Set-Location $repo

function Configure-R8 {
    param(
        [string]$Attempt,
        [string]$Name,
        [int]$Epochs,
        [int]$Sweeps,
        [string]$Beta,
        [string]$Lr,
        [string]$LrBias,
        [string]$ResumeCheckpoint
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
    $env:ISING_MNIST_PM_FREE_SWEEPS = "$Sweeps"
    $env:ISING_MNIST_PM_NUDGE_SWEEPS = "$Sweeps"
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
    $env:ISING_MNIST_PM_SEED = "2468"
    $env:ISING_MNIST_PM_PROGRESS = "true"
    $env:ISING_MNIST_PM_PROGRESS_BAR = "false"
    $env:ISING_MNIST_PM_PROGRESS_EVERY = "5"
    $env:ISING_MNIST_PM_RESUME_CHECKPOINT = $ResumeCheckpoint

    return $outdir
}

function Run-R8 {
    param(
        [string]$Attempt,
        [string]$Name,
        [int]$Epochs,
        [int]$Sweeps,
        [string]$Beta,
        [string]$Lr,
        [string]$LrBias,
        [string]$ResumeCheckpoint = ""
    )

    $outdir = Configure-R8 -Attempt $Attempt -Name $Name -Epochs $Epochs -Sweeps $Sweeps -Beta $Beta -Lr $Lr -LrBias $LrBias -ResumeCheckpoint $ResumeCheckpoint
    $stdout = Join-Path $logdir "$Attempt`_stdout.log"
    $stderr = Join-Path $logdir "$Attempt`_stderr.log"

    Add-Content -Path $status -Value "[$(Get-Date -Format s)] START $Attempt outdir=$outdir resume=$ResumeCheckpoint"
    $proc = Start-Process -FilePath $julia -ArgumentList @("-t", "32", "--project=ext/IsingLearning", $script) -WorkingDirectory $repo -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
    Add-Content -Path $status -Value "[$(Get-Date -Format s)] PID $Attempt $($proc.Id)"
    Wait-Process -Id $proc.Id
    Add-Content -Path $status -Value "[$(Get-Date -Format s)] END $Attempt"
}

Run-R8 -Attempt "a15r" -Name "r8_a15_resume" -Epochs 76 -Sweeps 50 -Beta "1.0" -Lr "0.00001" -LrBias "0.000001" -ResumeCheckpoint $a15Checkpoint
Run-R8 -Attempt "a16" -Name "r8_s50_b075_lr10" -Epochs 200 -Sweeps 50 -Beta "0.75" -Lr "0.00001" -LrBias "0.000001"

Add-Content -Path $status -Value "[$(Get-Date -Format s)] CONTINUATION COMPLETE"
