$ErrorActionPreference = "Stop"

$diag = Split-Path -Parent $MyInvocation.MyCommand.Path
$script = Join-Path $diag "low_beta_stateavg_training.jl"
$root = Join-Path $diag "tangent_observable_grid"
$logs = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $root, $logs | Out-Null

$repo = $diag
while (-not (Test-Path (Join-Path $repo "ext\IsingLearning\Project.toml"))) {
    $parent = Split-Path -Parent $repo
    if ($parent -eq $repo) {
        throw "Could not find repository root from $diag"
    }
    $repo = $parent
}
Set-Location $repo

function Format-Token {
    param([Parameter(Mandatory = $true)][string]$Value)
    return ($Value -replace "\.", "p" -replace "-", "m")
}

function Invoke-TangentObservableRun {
    param(
        [Parameter(Mandatory = $true)][string]$Beta,
        [Parameter(Mandatory = $true)][string]$LearningRate,
        [Parameter(Mandatory = $true)][string]$FreeBurn,
        [Parameter(Mandatory = $true)][string]$FreeAvg,
        [Parameter(Mandatory = $true)][string]$NudgeBurn,
        [Parameter(Mandatory = $true)][string]$NudgeAvg
    )

    $name = "r8_tangent_obs_b$(Format-Token $Beta)_free${FreeBurn}p${FreeAvg}_nudge${NudgeBurn}p${NudgeAvg}_lr$(Format-Token $LearningRate)_e80"
    $outdir = Join-Path $root $name
    $env:ISING_MNIST_PM_NAME = $name
    $env:ISING_MNIST_PM_OUTDIR = $outdir
    $env:ISING_MNIST_PM_RADII = "8"
    $env:ISING_MNIST_PM_RADIUS = "8"
    $env:ISING_MNIST_PM_WORKERS = "16"
    $env:ISING_MNIST_PM_EPOCHS = "80"
    $env:ISING_MNIST_PM_BATCHSIZE = "100"
    $env:ISING_MNIST_PM_JOB_CHUNK_SIZE = "1"
    $env:ISING_MNIST_PM_TRAIN_PER_CLASS = "50"
    $env:ISING_MNIST_PM_TEST_PER_CLASS = "20"
    $env:ISING_MNIST_PM_FREE_READS = "1"
    $env:ISING_MNIST_PM_NUDGE_READS = "1"
    $env:ISING_MNIST_PM_FREE_SWEEPS = $FreeBurn
    $env:ISING_MNIST_PM_NUDGE_SWEEPS = $NudgeBurn
    $env:ISING_MNIST_PM_BETA = $Beta
    $env:ISING_MNIST_PM_HOT_TEMP = "5.0"
    $env:ISING_MNIST_PM_COLD_TEMP = "0.003"
    $env:ISING_MNIST_PM_REVERSE_TEMP = "0.3"
    $env:ISING_MNIST_PM_LR_W0 = $LearningRate
    $env:ISING_MNIST_PM_LR_W12 = $LearningRate
    $env:ISING_MNIST_PM_LR_W2O = $LearningRate
    $env:ISING_MNIST_PM_LR_W11 = "0.0"
    $env:ISING_MNIST_PM_LR_W22 = "0.0"
    $env:ISING_MNIST_PM_LR_WOO = "0.0"
    $env:ISING_MNIST_PM_LR_B = "0.000001"
    $env:ISING_MNIST_PM_PROGRESS = "true"
    $env:ISING_MNIST_PM_PROGRESS_BAR = "false"
    $env:ISING_MNIST_PM_PROGRESS_EVERY = "5"
    $env:ISING_MNIST_PM_RESUME_CHECKPOINT = ""
    $env:ISING_MNIST_STATEAVG_ESTIMATOR = "observable_symmetric"
    $env:ISING_MNIST_TANGENT_NUDGE = "true"
    $env:ISING_MNIST_STATEAVG_FREE_BURNIN_SWEEPS = $FreeBurn
    $env:ISING_MNIST_STATEAVG_FREE_AVERAGE_SWEEPS = $FreeAvg
    $env:ISING_MNIST_STATEAVG_NUDGE_BURNIN_SWEEPS = $NudgeBurn
    $env:ISING_MNIST_STATEAVG_NUDGE_AVERAGE_SWEEPS = $NudgeAvg
    $env:ISING_MNIST_STATEAVG_SAMPLE_EVERY_SWEEPS = "1"

    $manifest = Join-Path $root "manifest.csv"
    if (-not (Test-Path $manifest)) {
        "timestamp,name,beta,lr_w,free_burn,free_avg,nudge_burn,nudge_avg,cold_T,reverse_T,outdir" | Out-File -FilePath $manifest -Encoding utf8
    }
    "$(Get-Date -Format s),$name,$Beta,$LearningRate,$FreeBurn,$FreeAvg,$NudgeBurn,$NudgeAvg,0.003,0.3,$outdir" | Out-File -FilePath $manifest -Append -Encoding utf8

    $stdout = Join-Path $logs "$($name)_stdout.log"
    $stderr = Join-Path $logs "$($name)_stderr.log"
    "[$(Get-Date -Format s)] START $name beta=$Beta lr=$LearningRate free=$FreeBurn/$FreeAvg nudge=$NudgeBurn/$NudgeAvg" | Tee-Object -FilePath $stdout -Append
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    julia -t 16 --project=ext/IsingLearning $script 1>> $stdout 2>> $stderr
    $exit = $LASTEXITCODE
    $ErrorActionPreference = $oldErrorActionPreference
    "[$(Get-Date -Format s)] END $name exit=$exit" | Tee-Object -FilePath $stdout -Append
    if ($exit -ne 0) {
        throw "Tangent observable run $name failed with exit code $exit"
    }
}

$runs = @(
    @{ Beta = "3.0";  LearningRate = "0.00001";  FreeBurn = "10"; FreeAvg = "3"; NudgeBurn = "50"; NudgeAvg = "25" },
    @{ Beta = "5.0";  LearningRate = "0.00001";  FreeBurn = "10"; FreeAvg = "3"; NudgeBurn = "50"; NudgeAvg = "25" },
    @{ Beta = "10.0"; LearningRate = "0.00001";  FreeBurn = "10"; FreeAvg = "3"; NudgeBurn = "50"; NudgeAvg = "25" },
    @{ Beta = "5.0";  LearningRate = "0.000005"; FreeBurn = "5";  FreeAvg = "1"; NudgeBurn = "75"; NudgeAvg = "50" }
)

foreach ($run in $runs) {
    Invoke-TangentObservableRun `
        -Beta $run.Beta `
        -LearningRate $run.LearningRate `
        -FreeBurn $run.FreeBurn `
        -FreeAvg $run.FreeAvg `
        -NudgeBurn $run.NudgeBurn `
        -NudgeAvg $run.NudgeAvg
}
