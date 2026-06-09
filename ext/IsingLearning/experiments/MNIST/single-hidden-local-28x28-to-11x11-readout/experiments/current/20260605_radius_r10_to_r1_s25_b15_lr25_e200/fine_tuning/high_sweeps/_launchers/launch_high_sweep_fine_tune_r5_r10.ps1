$ErrorActionPreference = "Stop"

$repo = "C:\Users\fenje\dev\InteractiveIsing.jl"
$julia = "C:\Users\fenje\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe"
$script = "ext/IsingLearning/experiments/MNIST/single-hidden-local-28x28-to-11x11-readout/mnist_local_manager_grid.jl"
$series = Join-Path $repo "ext\IsingLearning\experiments\MNIST\single-hidden-local-28x28-to-11x11-readout\experiments\current\20260605_radius_r10_to_r1_s25_b15_lr25_e200"
$high = Join-Path $series "fine_tuning\high_sweeps"
$logdir = Join-Path $high "logs"
$status = Join-Path $logdir "status.log"

New-Item -ItemType Directory -Force -Path $logdir | Out-Null
Set-Location $repo

$baseRuns = @(
    @{
        Radius = 5
        SourceEpoch = 138
        SourceCheckpoint = Join-Path $series "r5\best_params.bin"
    },
    @{
        Radius = 10
        SourceEpoch = 151
        SourceCheckpoint = Join-Path $series "r10\best_params.bin"
    }
)

$branches = @(
    @{
        Name = "s50_beta15_lr10"
        Sweeps = 50
        Beta = "1.5"
        Lr = "0.00001"
        LrBias = "0.000001"
        ExtraEpochs = 100
    },
    @{
        Name = "s50_beta10_lr10"
        Sweeps = 50
        Beta = "1.0"
        Lr = "0.00001"
        LrBias = "0.000001"
        ExtraEpochs = 100
    },
    @{
        Name = "s75_beta10_lr5"
        Sweeps = 75
        Beta = "1.0"
        Lr = "0.000005"
        LrBias = "0.0000005"
        ExtraEpochs = 80
    }
)

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

function Configure-FineTune {
    param(
        [int]$Radius,
        [int]$TargetEpochs,
        [string]$OutRoot,
        [int]$Sweeps,
        [string]$Beta,
        [string]$Lr,
        [string]$LrBias,
        [string]$ResumeCheckpoint
    )

    $env:ISING_MNIST_PM_NAME = "r$Radius"
    $env:ISING_MNIST_PM_OUTDIR = $OutRoot
    $env:ISING_MNIST_PM_RADII = "$Radius"
    $env:ISING_MNIST_PM_RADIUS = "$Radius"
    $env:ISING_MNIST_PM_WORKERS = "16"
    $env:ISING_MNIST_PM_EPOCHS = "$TargetEpochs"
    $env:ISING_MNIST_PM_BATCHSIZE = "128"
    $env:ISING_MNIST_PM_JOB_CHUNK_SIZE = "8"
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
}

foreach ($branch in $branches) {
    foreach ($baseRun in $baseRuns) {
        $radius = [int]$baseRun.Radius
        $sourceEpoch = [int]$baseRun.SourceEpoch
        $targetEpochs = $sourceEpoch + [int]$branch.ExtraEpochs
        $sourceCheckpoint = [string]$baseRun.SourceCheckpoint
        $outRoot = Join-Path $high $branch.Name
        $runDir = Join-Path $outRoot "r$radius"
        $metrics = Join-Path $runDir "metrics.csv"
        $latest = Join-Path $runDir "latest_checkpoint.bin"
        $lastEpoch = Get-LastMetricEpoch -MetricsPath $metrics

        if ($lastEpoch -ge $targetEpochs) {
            Add-Content -Path $status -Value "[$(Get-Date -Format s)] SKIP $($branch.Name) r$radius complete last_epoch=$lastEpoch"
            continue
        }

        if ($lastEpoch -ge 0) {
            if (!(Test-Path $latest)) {
                Add-Content -Path $status -Value "[$(Get-Date -Format s)] BLOCK $($branch.Name) r$radius has metrics epoch=$lastEpoch but no checkpoint=$latest"
                throw "Cannot resume $($branch.Name) r$radius without latest checkpoint"
            }
            $resume = $latest
        } else {
            if (!(Test-Path $sourceCheckpoint)) {
                throw "Missing source checkpoint: $sourceCheckpoint"
            }
            $resume = $sourceCheckpoint
            New-Item -ItemType Directory -Force -Path $runDir | Out-Null
            Copy-Item -Path $sourceCheckpoint -Destination (Join-Path $runDir "initial_best_params.bin") -Force
        }

        Configure-FineTune `
            -Radius $radius `
            -TargetEpochs $targetEpochs `
            -OutRoot $outRoot `
            -Sweeps ([int]$branch.Sweeps) `
            -Beta ([string]$branch.Beta) `
            -Lr ([string]$branch.Lr) `
            -LrBias ([string]$branch.LrBias) `
            -ResumeCheckpoint $resume

        $label = "$($branch.Name)_r$radius"
        $stdout = Join-Path $logdir "$label`_stdout.log"
        $stderr = Join-Path $logdir "$label`_stderr.log"

        Add-Content -Path $status -Value "[$(Get-Date -Format s)] START $label last_epoch=$lastEpoch target=$targetEpochs resume=$resume"
        $proc = Start-Process -FilePath $julia -ArgumentList @("-t", "32", "--project=ext/IsingLearning", $script) -WorkingDirectory $repo -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
        Add-Content -Path $status -Value "[$(Get-Date -Format s)] PID $label $($proc.Id)"
        Wait-Process -Id $proc.Id

        $exitCode = $proc.ExitCode
        Add-Content -Path $status -Value "[$(Get-Date -Format s)] END $label exit=$exitCode"
        if ($exitCode -ne 0) {
            throw "$label failed with exit code $exitCode"
        }
    }
}

Add-Content -Path $status -Value "[$(Get-Date -Format s)] HIGH-SWEEP FINE TUNING COMPLETE"
