$ErrorActionPreference = "Stop"

$repo = "C:\Users\fenje\dev\InteractiveIsing.jl"
$arch = Join-Path $repo "ext\IsingLearning\experiments\MNIST\single-hidden-local-28x28-to-11x11-readout"
$current = Join-Path $arch "experiments\current"
$control = Join-Path $arch "diagnostics\runs\20260528_120724_r8_dynamics_compare_from_scratch_e10"
$summaryPath = Join-Path $control "summary.csv"

$configs = @(
    @{ label = "metropolis"; dynamics = "metropolis"; stepsize = ""; adjusted = ""; folder = "20260528_120724_mnist_single_hidden_local_r8_metropolis_from_scratch_s50_e10_32w" },
    @{ label = "local_langevin_ss003"; dynamics = "local_langevin"; stepsize = "0.03"; adjusted = "false"; folder = "20260528_120724_mnist_single_hidden_local_r8_local_langevin_ss003_from_scratch_s50_e10_32w" },
    @{ label = "local_langevin_ss010"; dynamics = "local_langevin"; stepsize = "0.10"; adjusted = "false"; folder = "20260528_120724_mnist_single_hidden_local_r8_local_langevin_ss010_from_scratch_s50_e10_32w" },
    @{ label = "local_langevin_ss030"; dynamics = "local_langevin"; stepsize = "0.30"; adjusted = "false"; folder = "20260528_120724_mnist_single_hidden_local_r8_local_langevin_ss030_from_scratch_s50_e10_32w" },
    @{ label = "global_langevin_ss003"; dynamics = "global_langevin"; stepsize = "0.03"; adjusted = "false"; folder = "20260528_120724_mnist_single_hidden_local_r8_global_langevin_ss003_from_scratch_s50_e10_32w" },
    @{ label = "global_langevin_ss010"; dynamics = "global_langevin"; stepsize = "0.10"; adjusted = "false"; folder = "20260528_120724_mnist_single_hidden_local_r8_global_langevin_ss010_from_scratch_s50_e10_32w" },
    @{ label = "global_langevin_ss030"; dynamics = "global_langevin"; stepsize = "0.30"; adjusted = "false"; folder = "20260528_120724_mnist_single_hidden_local_r8_global_langevin_ss030_from_scratch_s50_e10_32w" }
)

$juliaCode = @'
include(raw"C:\Users\fenje\dev\InteractiveIsing.jl\ext\IsingLearning\experiments\MNIST\single-hidden-local-28x28-to-11x11-readout\mnist_local_manager_grid.jl")

config = LocalMNISTManagerConfig(;
    name = ENV["ISING_MNIST_PM_NAME"],
    local_radius = 8,
    outdir = ENV["ISING_MNIST_PM_OUTDIR"],
)
run_config!(config)
'@

$plotCode = @'
using CairoMakie

summary_path = ARGS[1]
out_path = ARGS[2]
lines = readlines(summary_path)
isempty(lines) && error("summary is empty")
clean(value) = strip(strip(value), '"')
header = clean.(split(lines[1], ","))
rows = NamedTuple[]
for line in lines[2:end]
    isempty(strip(line)) && continue
    values = clean.(split(line, ","))
    push!(rows, (; zip(Symbol.(header), values)...))
end

labels = [row.label for row in rows]
best = [parse(Float64, row.best_test_accuracy) for row in rows]
final = [parse(Float64, row.final_test_accuracy) for row in rows]

fig = Figure(size = (1200, 720))
ax = Axis(fig[1, 1], xlabel = "dynamics", ylabel = "test accuracy", title = "R8 dynamics from scratch, 10 epochs")
x = 1:length(rows)
barplot!(ax, x .- 0.18, best, width = 0.34, color = :steelblue, label = "best")
barplot!(ax, x .+ 0.18, final, width = 0.34, color = :orange, label = "final")
ax.xticks = (collect(x), labels)
ax.xticklabelrotation = pi / 5
axislegend(ax, position = :lt)
save(out_path, fig)
'@

function Write-SummaryRow {
    param(
        [hashtable]$Cfg,
        [string]$OutDir,
        [int]$ExitCode
    )

    $metrics = Join-Path $OutDir "metrics.csv"
    if (Test-Path -LiteralPath $metrics) {
        $rows = Import-Csv -LiteralPath $metrics
        $last = $rows | Select-Object -Last 1
        $best = $rows | Sort-Object { [double]$_.test_accuracy } -Descending | Select-Object -First 1
        $record = [pscustomobject]@{
            label = $Cfg.label
            dynamics = $Cfg.dynamics
            stepsize = $Cfg.stepsize
            adjusted = $Cfg.adjusted
            run_dir = $OutDir
            exit_code = $ExitCode
            best_epoch = $best.epoch
            best_test_accuracy = $best.test_accuracy
            final_epoch = $last.epoch
            final_train_accuracy = $last.train_accuracy
            final_test_accuracy = $last.test_accuracy
            final_skipped = $last.skipped
        }
    } else {
        $record = [pscustomobject]@{
            label = $Cfg.label
            dynamics = $Cfg.dynamics
            stepsize = $Cfg.stepsize
            adjusted = $Cfg.adjusted
            run_dir = $OutDir
            exit_code = $ExitCode
            best_epoch = ""
            best_test_accuracy = ""
            final_epoch = ""
            final_train_accuracy = ""
            final_test_accuracy = ""
            final_skipped = ""
        }
    }

    if (Test-Path -LiteralPath $summaryPath) {
        $record | Export-Csv -LiteralPath $summaryPath -NoTypeInformation -Append
    } else {
        $record | Export-Csv -LiteralPath $summaryPath -NoTypeInformation
    }
}

Push-Location $repo
try {
    Remove-Item -LiteralPath $summaryPath -Force -ErrorAction SilentlyContinue
    Remove-Item Env:\ISING_MNIST_PM_RESUME_CHECKPOINT -ErrorAction SilentlyContinue

    foreach ($cfg in $configs) {
        $outdir = Join-Path $current $cfg.folder
        New-Item -ItemType Directory -Force -Path $outdir | Out-Null

        $env:ISING_MNIST_PM_NAME = $cfg.label
        $env:ISING_MNIST_PM_WORKERS = "32"
        $env:ISING_MNIST_PM_EPOCHS = "10"
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
        $env:ISING_MNIST_PM_OUTDIR = $outdir
        $env:ISING_MNIST_PM_DYNAMICS = $cfg.dynamics
        $env:ISING_MNIST_PM_PROGRESS = "true"
        $env:ISING_MNIST_PM_PROGRESS_EVERY = "10"
        Remove-Item Env:\ISING_MNIST_PM_RESUME_CHECKPOINT -ErrorAction SilentlyContinue

        if ($cfg.stepsize -ne "") {
            $env:ISING_MNIST_PM_LANGEVIN_STEPSIZE = $cfg.stepsize
            $env:ISING_MNIST_PM_LANGEVIN_ADJUSTED = $cfg.adjusted
        } else {
            Remove-Item Env:\ISING_MNIST_PM_LANGEVIN_STEPSIZE -ErrorAction SilentlyContinue
            Remove-Item Env:\ISING_MNIST_PM_LANGEVIN_ADJUSTED -ErrorAction SilentlyContinue
        }

        $stdout = Join-Path $outdir "stdout.log"
        $stderr = Join-Path $outdir "stderr.log"
        $started = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
        "[$started] starting $($cfg.label) from scratch dynamics=$($cfg.dynamics) stepsize=$($cfg.stepsize) outdir=$outdir" | Set-Content -Encoding UTF8 -Path $stdout

        & julia -t 32 --project=ext\IsingLearning -e $juliaCode 1>> $stdout 2> $stderr
        $exit = $LASTEXITCODE
        $finished = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
        "[$finished] finished $($cfg.label) exit=$exit" | Add-Content -Encoding UTF8 -Path $stdout

        Add-Content -Encoding UTF8 -Path (Join-Path $outdir "settings.md") -Value @(
            "",
            "## Dynamics",
            "",
            "- dynamics: $($cfg.dynamics)",
            "- langevin stepsize: $($cfg.stepsize)",
            "- langevin adjusted: $($cfg.adjusted)",
            "- initialized from: scratch",
            "- comparison control folder: $control"
        )

        Write-SummaryRow -Cfg $cfg -OutDir $outdir -ExitCode $exit
        if ($exit -ne 0) {
            throw "$($cfg.label) failed with exit code $exit"
        }
    }

    & julia --project=ext\IsingLearning -e $plotCode $summaryPath (Join-Path $control "summary.png")
    if ($LASTEXITCODE -ne 0) {
        throw "summary plot failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
