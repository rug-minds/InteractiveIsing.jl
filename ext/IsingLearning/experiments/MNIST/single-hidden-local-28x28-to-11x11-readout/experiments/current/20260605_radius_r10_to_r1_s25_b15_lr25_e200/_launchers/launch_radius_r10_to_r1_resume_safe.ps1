$ErrorActionPreference = "Stop"

$repo = "C:\Users\fenje\dev\InteractiveIsing.jl"
$julia = "C:\Users\fenje\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe"
$script = "ext/IsingLearning/experiments/MNIST/single-hidden-local-28x28-to-11x11-readout/mnist_local_manager_grid.jl"
$series = Join-Path $repo "ext\IsingLearning\experiments\MNIST\single-hidden-local-28x28-to-11x11-readout\experiments\current\20260605_radius_r10_to_r1_s25_b15_lr25_e200"
$logdir = Join-Path $series "logs"
$status = Join-Path $logdir "status.log"
$targetEpochs = 200
$radii = @(10, 9, 8, 7, 6, 5, 4, 3, 2, 1)

New-Item -ItemType Directory -Force -Path $logdir | Out-Null
Set-Location $repo

function Get-LastMetricEpoch {
    param([string]$MetricsPath)

    if (!(Test-Path $MetricsPath)) {
        return -1
    }

    $last = Get-Content -Path $MetricsPath | Where-Object { $_.Trim().Length -gt 0 } | Select-Object -Last 1
    if ($null -eq $last -or $last.StartsWith("timestamp,")) {
        return -1
    }

    $cells = $last.Split(",")
    if ($cells.Count -lt 2) {
        return -1
    }

    return [int]$cells[1]
}

function Configure-Radius {
    param([int]$Radius, [string]$ResumeCheckpoint)

    $env:ISING_MNIST_PM_NAME = "r$Radius"
    $env:ISING_MNIST_PM_OUTDIR = $series
    $env:ISING_MNIST_PM_RADII = "$Radius"
    $env:ISING_MNIST_PM_RADIUS = "$Radius"
    $env:ISING_MNIST_PM_WORKERS = "16"
    $env:ISING_MNIST_PM_EPOCHS = "$targetEpochs"
    $env:ISING_MNIST_PM_BATCHSIZE = "128"
    $env:ISING_MNIST_PM_JOB_CHUNK_SIZE = "8"
    $env:ISING_MNIST_PM_TRAIN_PER_CLASS = "100"
    $env:ISING_MNIST_PM_TEST_PER_CLASS = "20"
    $env:ISING_MNIST_PM_FREE_SWEEPS = "25"
    $env:ISING_MNIST_PM_NUDGE_SWEEPS = "25"
    $env:ISING_MNIST_PM_FREE_READS = "3"
    $env:ISING_MNIST_PM_NUDGE_READS = "3"
    $env:ISING_MNIST_PM_BETA = "1.5"
    $env:ISING_MNIST_PM_LR_W0 = "0.000025"
    $env:ISING_MNIST_PM_LR_W12 = "0.000025"
    $env:ISING_MNIST_PM_LR_W2O = "0.000025"
    $env:ISING_MNIST_PM_LR_B = "0.0000025"
    $env:ISING_MNIST_PM_TRAIN_OUTPUT_BIAS = "false"
    $env:ISING_MNIST_PM_TRAIN_INTERNAL = "false"
    $env:ISING_MNIST_PM_DYNAMICS = "metropolis"
    $env:ISING_MNIST_PM_SEED = "2468"
    $env:ISING_MNIST_PM_PROGRESS = "true"
    $env:ISING_MNIST_PM_PROGRESS_BAR = "false"
    $env:ISING_MNIST_PM_PROGRESS_EVERY = "5"
    $env:ISING_MNIST_PM_RESUME_CHECKPOINT = $ResumeCheckpoint
}

foreach ($radius in $radii) {
    $runDir = Join-Path $series "r$radius"
    $metrics = Join-Path $runDir "metrics.csv"
    $checkpoint = Join-Path $runDir "latest_checkpoint.bin"
    $lastEpoch = Get-LastMetricEpoch -MetricsPath $metrics

    if ($lastEpoch -ge $targetEpochs) {
        Add-Content -Path $status -Value "[$(Get-Date -Format s)] SKIP r$radius complete last_epoch=$lastEpoch"
        continue
    }

    $resume = ""
    if ($lastEpoch -ge 0) {
        if (!(Test-Path $checkpoint)) {
            Add-Content -Path $status -Value "[$(Get-Date -Format s)] BLOCK r$radius has metrics epoch=$lastEpoch but no checkpoint=$checkpoint"
            throw "Cannot resume r$radius without latest_checkpoint.bin"
        }
        $resume = $checkpoint
    }

    Configure-Radius -Radius $radius -ResumeCheckpoint $resume
    $stdout = Join-Path $logdir "r$radius`_stdout.log"
    $stderr = Join-Path $logdir "r$radius`_stderr.log"

    Add-Content -Path $status -Value "[$(Get-Date -Format s)] START r$radius last_epoch=$lastEpoch target=$targetEpochs resume=$resume"
    $proc = Start-Process -FilePath $julia -ArgumentList @("-t", "32", "--project=ext/IsingLearning", $script) -WorkingDirectory $repo -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
    Add-Content -Path $status -Value "[$(Get-Date -Format s)] PID r$radius $($proc.Id)"
    Wait-Process -Id $proc.Id

    $exitCode = $proc.ExitCode
    Add-Content -Path $status -Value "[$(Get-Date -Format s)] END r$radius exit=$exitCode"
    if ($exitCode -ne 0) {
        throw "r$radius failed with exit code $exitCode"
    }
}

Add-Content -Path $status -Value "[$(Get-Date -Format s)] RADIUS GRID COMPLETE"
