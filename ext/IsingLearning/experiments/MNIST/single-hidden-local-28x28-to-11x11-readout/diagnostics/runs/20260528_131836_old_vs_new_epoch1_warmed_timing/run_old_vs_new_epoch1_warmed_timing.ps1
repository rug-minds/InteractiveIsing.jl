$ErrorActionPreference = "Stop"

$repo = "C:\Users\fenje\dev\InteractiveIsing.jl"
$arch = Join-Path $repo "ext\IsingLearning\experiments\MNIST\single-hidden-local-28x28-to-11x11-readout"
$control = Join-Path $arch "diagnostics\runs\20260528_131836_old_vs_new_epoch1_warmed_timing"
$current = Join-Path $arch "experiments\current"
$summary = Join-Path $control "summary.csv"

$runs = @(
    @{
        label = "old"
        file = Join-Path $arch "mnist_local_manager_grid_OLD.jl"
    },
    @{
        label = "new"
        file = Join-Path $arch "mnist_local_manager_grid.jl"
    }
)

function Write-SummaryRow {
    param(
        [hashtable]$Run,
        [string]$Phase,
        [string]$Outdir,
        [double]$ElapsedSeconds,
        [int]$ExitCode
    )

    $metrics = Join-Path $Outdir "metrics.csv"
    if (Test-Path -LiteralPath $metrics) {
        $rows = Import-Csv -LiteralPath $metrics
        $last = $rows | Select-Object -Last 1
        $epoch1 = $rows | Where-Object { $_.epoch -eq "1" } | Select-Object -First 1
        $record = [pscustomobject]@{
            label = $Run.label
            phase = $Phase
            run_elapsed_seconds = [math]::Round($ElapsedSeconds, 3)
            exit_code = $ExitCode
            rows = $rows.Count
            final_epoch = $last.epoch
            epoch1_train_seconds = $epoch1.seconds
            epoch1_train_accuracy = $epoch1.train_accuracy
            epoch1_test_accuracy = $epoch1.test_accuracy
            epoch1_skipped = $epoch1.skipped
            run_dir = $Outdir
        }
    } else {
        $record = [pscustomobject]@{
            label = $Run.label
            phase = $Phase
            run_elapsed_seconds = [math]::Round($ElapsedSeconds, 3)
            exit_code = $ExitCode
            rows = 0
            final_epoch = ""
            epoch1_train_seconds = ""
            epoch1_train_accuracy = ""
            epoch1_test_accuracy = ""
            epoch1_skipped = ""
            run_dir = $Outdir
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
        $warmupOutdir = Join-Path $current "20260528_131836_$($run.label)_epoch1_32w_warmup"
        $measuredOutdir = Join-Path $current "20260528_131836_$($run.label)_epoch1_32w_warmed_timing"
        New-Item -ItemType Directory -Force -Path $warmupOutdir | Out-Null
        New-Item -ItemType Directory -Force -Path $measuredOutdir | Out-Null

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
        $env:ISING_MNIST_PM_PROGRESS = "true"
        $env:ISING_MNIST_PM_PROGRESS_BAR = "false"
        $env:ISING_MNIST_PM_PROGRESS_EVERY = "10"
        Remove-Item Env:\ISING_MNIST_PM_RESUME_CHECKPOINT -ErrorAction SilentlyContinue
        Remove-Item Env:\ISING_MNIST_PM_LANGEVIN_STEPSIZE -ErrorAction SilentlyContinue
        Remove-Item Env:\ISING_MNIST_PM_LANGEVIN_ADJUSTED -ErrorAction SilentlyContinue

        $stdout = Join-Path $control "$($run.label)_stdout.log"
        $stderr = Join-Path $control "$($run.label)_stderr.log"
        "[$(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss')] starting $($run.label) warmed diagnostic file=$($run.file)" | Set-Content -Encoding UTF8 -Path $stdout

        $code = @"
include(raw"$($run.file)")

function run_timed_phase!(phase, outdir)
    ENV["ISING_MNIST_PM_NAME"] = "$($run.label)_" * phase * "_epoch1_32w"
    ENV["ISING_MNIST_PM_OUTDIR"] = outdir
    elapsed = @elapsed begin
        config = LocalMNISTManagerConfig(;
            name = ENV["ISING_MNIST_PM_NAME"],
            local_radius = 8,
            outdir = ENV["ISING_MNIST_PM_OUTDIR"],
        )
        run_config!(config)
    end
    open(joinpath(outdir, "run_elapsed_seconds.txt"), "w") do io
        println(io, round(elapsed; digits = 3))
    end
    return elapsed
end

warmup_elapsed = run_timed_phase!("warmup", raw"$warmupOutdir")
measured_elapsed = run_timed_phase!("warmed_timing", raw"$measuredOutdir")
println("warmup_elapsed_seconds=", round(warmup_elapsed; digits = 3))
println("measured_elapsed_seconds=", round(measured_elapsed; digits = 3))
"@

        & julia -t 32 --project=ext\IsingLearning -e $code 1>> $stdout 2> $stderr
        $exit = $LASTEXITCODE

        $warmupElapsedPath = Join-Path $warmupOutdir "run_elapsed_seconds.txt"
        $measuredElapsedPath = Join-Path $measuredOutdir "run_elapsed_seconds.txt"
        $warmupElapsed = if (Test-Path -LiteralPath $warmupElapsedPath) { [double](Get-Content -LiteralPath $warmupElapsedPath -Raw) } else { 0.0 }
        $measuredElapsed = if (Test-Path -LiteralPath $measuredElapsedPath) { [double](Get-Content -LiteralPath $measuredElapsedPath -Raw) } else { 0.0 }

        "[$(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss')] finished $($run.label) warmed diagnostic exit=$exit measured_elapsed_seconds=$measuredElapsed" | Add-Content -Encoding UTF8 -Path $stdout
        Write-SummaryRow -Run $run -Phase "warmup" -Outdir $warmupOutdir -ElapsedSeconds $warmupElapsed -ExitCode $exit
        Write-SummaryRow -Run $run -Phase "warmed_timing" -Outdir $measuredOutdir -ElapsedSeconds $measuredElapsed -ExitCode $exit
        if ($exit -ne 0) {
            throw "$($run.label) warmed timing failed with exit code $exit"
        }
    }
}
finally {
    Pop-Location
}
