$ErrorActionPreference = "Stop"

$repo = "C:\Users\fenje\dev\InteractiveIsing.jl"
$arch = Join-Path $repo "ext\IsingLearning\experiments\MNIST\single-hidden-local-28x28-to-11x11-readout"
$control = Join-Path $arch "diagnostics\runs\20260528_125338_old_vs_new_epoch1_timing"
$current = Join-Path $arch "experiments\current"
$summary = Join-Path $control "summary.csv"

$runs = @(
    @{
        label = "old"
        file = Join-Path $arch "mnist_local_manager_grid_OLD.jl"
        outdir = Join-Path $current "20260528_125338_old_epoch1_32w_timing"
    },
    @{
        label = "new"
        file = Join-Path $arch "mnist_local_manager_grid.jl"
        outdir = Join-Path $current "20260528_125338_new_epoch1_32w_timing"
    }
)

function Write-SummaryRow {
    param(
        [hashtable]$Run,
        [double]$WallSeconds,
        [int]$ExitCode
    )

    $metrics = Join-Path $Run.outdir "metrics.csv"
    if (Test-Path -LiteralPath $metrics) {
        $rows = Import-Csv -LiteralPath $metrics
        $last = $rows | Select-Object -Last 1
        $epoch1 = $rows | Where-Object { $_.epoch -eq "1" } | Select-Object -First 1
        $record = [pscustomobject]@{
            label = $Run.label
            wall_seconds = [math]::Round($WallSeconds, 3)
            exit_code = $ExitCode
            rows = $rows.Count
            final_epoch = $last.epoch
            epoch1_train_seconds = $epoch1.seconds
            epoch1_train_accuracy = $epoch1.train_accuracy
            epoch1_test_accuracy = $epoch1.test_accuracy
            epoch1_skipped = $epoch1.skipped
            run_dir = $Run.outdir
        }
    } else {
        $record = [pscustomobject]@{
            label = $Run.label
            wall_seconds = [math]::Round($WallSeconds, 3)
            exit_code = $ExitCode
            rows = 0
            final_epoch = ""
            epoch1_train_seconds = ""
            epoch1_train_accuracy = ""
            epoch1_test_accuracy = ""
            epoch1_skipped = ""
            run_dir = $Run.outdir
        }
    }

    if (Test-Path -LiteralPath $summary) {
        $record | Export-Csv -LiteralPath $summary -NoTypeInformation -Append
    } else {
        $record | Export-Csv -LiteralPath $summary -NoTypeInformation
    }
}

Push-Location $repo
try {
    Remove-Item -LiteralPath $summary -Force -ErrorAction SilentlyContinue

    foreach ($run in $runs) {
        New-Item -ItemType Directory -Force -Path $run.outdir | Out-Null

        $env:ISING_MNIST_PM_NAME = "$($run.label)_epoch1_timing"
        $env:ISING_MNIST_PM_WORKERS = "32"
        $env:ISING_MNIST_PM_EPOCHS = "1"
        $env:ISING_MNIST_PM_BATCHSIZE = "32"
        $env:ISING_MNIST_PM_RADIUS = "8"
        $env:ISING_MNIST_PM_FREE_SWEEPS = "50"
        $env:ISING_MNIST_PM_NUDGE_SWEEPS = "50"
        $env:ISING_MNIST_PM_FREE_READS = "3"
        $env:ISING_MNIST_PM_NUDGE_READS = "3"
        $env:ISING_MNIST_PM_OPTIMIZER = "adam"
        $env:ISING_MNIST_PM_LR_W0 = "0.004"
        $env:ISING_MNIST_PM_LR_W12 = "0.004"
        $env:ISING_MNIST_PM_LR_W2O = "0.004"
        $env:ISING_MNIST_PM_LR_B = "0.0004"
        $env:ISING_MNIST_PM_GRADIENT_NORMALIZATION = "mean"
        $env:ISING_MNIST_PM_DYNAMICS = "metropolis"
        $env:ISING_MNIST_PM_OUTDIR = $run.outdir
        $env:ISING_MNIST_PM_PROGRESS = "true"
        $env:ISING_MNIST_PM_PROGRESS_BAR = "false"
        $env:ISING_MNIST_PM_PROGRESS_EVERY = "10"
        Remove-Item Env:\ISING_MNIST_PM_RESUME_CHECKPOINT -ErrorAction SilentlyContinue
        Remove-Item Env:\ISING_MNIST_PM_LANGEVIN_STEPSIZE -ErrorAction SilentlyContinue
        Remove-Item Env:\ISING_MNIST_PM_LANGEVIN_ADJUSTED -ErrorAction SilentlyContinue

        $stdout = Join-Path $run.outdir "stdout.log"
        $stderr = Join-Path $run.outdir "stderr.log"
        "[$(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss')] starting $($run.label) timing file=$($run.file)" | Set-Content -Encoding UTF8 -Path $stdout

        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        & julia -t 32 --project=ext\IsingLearning -e "include(raw`"$($run.file)`"); config = LocalMNISTManagerConfig(; name = ENV[`"ISING_MNIST_PM_NAME`"], local_radius = 8, outdir = ENV[`"ISING_MNIST_PM_OUTDIR`"]); run_config!(config)" 1>> $stdout 2> $stderr
        $exit = $LASTEXITCODE
        $sw.Stop()

        "[$(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss')] finished $($run.label) timing exit=$exit wall_seconds=$([math]::Round($sw.Elapsed.TotalSeconds, 3))" | Add-Content -Encoding UTF8 -Path $stdout
        Write-SummaryRow -Run $run -WallSeconds $sw.Elapsed.TotalSeconds -ExitCode $exit
        if ($exit -ne 0) {
            throw "$($run.label) timing failed with exit code $exit"
        }
    }
}
finally {
    Pop-Location
}
