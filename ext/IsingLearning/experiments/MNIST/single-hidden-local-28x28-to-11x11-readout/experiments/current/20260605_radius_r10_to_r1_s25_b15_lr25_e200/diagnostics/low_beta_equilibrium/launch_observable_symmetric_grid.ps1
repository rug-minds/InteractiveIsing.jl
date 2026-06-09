$ErrorActionPreference = "Stop"

$diag = Split-Path -Parent $MyInvocation.MyCommand.Path
$script = Join-Path $diag "low_beta_stateavg_training.jl"
$root = Join-Path $diag "observable_symmetric_grid"
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

function Invoke-ObservableSymmetricRun {
    param(
        [Parameter(Mandatory = $true)][string]$Beta,
        [Parameter(Mandatory = $true)][string]$ColdT,
        [Parameter(Mandatory = $true)][string]$ReverseT,
        [Parameter(Mandatory = $true)][string]$LearningRate,
        [Parameter(Mandatory = $true)][string]$BiasLearningRate,
        [Parameter(Mandatory = $true)][string]$AverageSweeps
    )

    $name = "r8_obs_sym_b$(Format-Token $Beta)_Tc$(Format-Token $ColdT)_Tr$(Format-Token $ReverseT)_lr$(Format-Token $LearningRate)_avg$($AverageSweeps)_e80"
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
    $env:ISING_MNIST_PM_FREE_SWEEPS = "25"
    $env:ISING_MNIST_PM_NUDGE_SWEEPS = "25"
    $env:ISING_MNIST_PM_BETA = $Beta
    $env:ISING_MNIST_PM_HOT_TEMP = "5.0"
    $env:ISING_MNIST_PM_COLD_TEMP = $ColdT
    $env:ISING_MNIST_PM_REVERSE_TEMP = $ReverseT
    $env:ISING_MNIST_PM_LR_W0 = $LearningRate
    $env:ISING_MNIST_PM_LR_W12 = $LearningRate
    $env:ISING_MNIST_PM_LR_W2O = $LearningRate
    $env:ISING_MNIST_PM_LR_W11 = "0.0"
    $env:ISING_MNIST_PM_LR_W22 = "0.0"
    $env:ISING_MNIST_PM_LR_WOO = "0.0"
    $env:ISING_MNIST_PM_LR_B = $BiasLearningRate
    $env:ISING_MNIST_PM_PROGRESS = "true"
    $env:ISING_MNIST_PM_PROGRESS_BAR = "false"
    $env:ISING_MNIST_PM_PROGRESS_EVERY = "5"
    $env:ISING_MNIST_PM_RESUME_CHECKPOINT = ""
    $env:ISING_MNIST_STATEAVG_ESTIMATOR = "observable_symmetric"
    $env:ISING_MNIST_STATEAVG_BURNIN_SWEEPS = "25"
    $env:ISING_MNIST_STATEAVG_AVERAGE_SWEEPS = $AverageSweeps
    $env:ISING_MNIST_STATEAVG_SAMPLE_EVERY_SWEEPS = "1"

    $stdout = Join-Path $logs "$($name)_stdout.log"
    $stderr = Join-Path $logs "$($name)_stderr.log"
    $manifest = Join-Path $root "manifest.csv"
    if (-not (Test-Path $manifest)) {
        "timestamp,name,beta,cold_T,reverse_T,lr_w,lr_b,average_sweeps,outdir" | Out-File -FilePath $manifest -Encoding utf8
    }
    "$(Get-Date -Format s),$name,$Beta,$ColdT,$ReverseT,$LearningRate,$BiasLearningRate,$AverageSweeps,$outdir" | Out-File -FilePath $manifest -Append -Encoding utf8

    "[$(Get-Date -Format s)] START $name beta=$Beta cold_T=$ColdT reverse_T=$ReverseT lr=$LearningRate avg=$AverageSweeps" | Tee-Object -FilePath $stdout -Append
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    julia -t 16 --project=ext/IsingLearning $script 1>> $stdout 2>> $stderr
    $exit = $LASTEXITCODE
    $ErrorActionPreference = $oldErrorActionPreference
    "[$(Get-Date -Format s)] END $name exit=$exit" | Tee-Object -FilePath $stdout -Append
    if ($exit -ne 0) {
        throw "Observable symmetric run $name failed with exit code $exit"
    }
}

$runs = @(
    @{ Beta = "0.5"; ColdT = "0.003"; ReverseT = "0.1"; LearningRate = "0.00001"; BiasLearningRate = "0.000001"; AverageSweeps = "10" },
    @{ Beta = "0.5"; ColdT = "0.001"; ReverseT = "0.1"; LearningRate = "0.00001"; BiasLearningRate = "0.000001"; AverageSweeps = "10" },
    @{ Beta = "1.0"; ColdT = "0.003"; ReverseT = "0.1"; LearningRate = "0.00001"; BiasLearningRate = "0.000001"; AverageSweeps = "10" },
    @{ Beta = "1.0"; ColdT = "0.001"; ReverseT = "0.1"; LearningRate = "0.00001"; BiasLearningRate = "0.000001"; AverageSweeps = "10" },
    @{ Beta = "1.0"; ColdT = "0.003"; ReverseT = "0.3"; LearningRate = "0.00001"; BiasLearningRate = "0.000001"; AverageSweeps = "10" },
    @{ Beta = "2.0"; ColdT = "0.001"; ReverseT = "0.1"; LearningRate = "0.00001"; BiasLearningRate = "0.000001"; AverageSweeps = "10" },
    @{ Beta = "2.0"; ColdT = "0.003"; ReverseT = "0.3"; LearningRate = "0.00001"; BiasLearningRate = "0.000001"; AverageSweeps = "10" },
    @{ Beta = "1.0"; ColdT = "0.001"; ReverseT = "0.05"; LearningRate = "0.000005"; BiasLearningRate = "0.0000005"; AverageSweeps = "20" }
)

foreach ($run in $runs) {
    Invoke-ObservableSymmetricRun `
        -Beta $run.Beta `
        -ColdT $run.ColdT `
        -ReverseT $run.ReverseT `
        -LearningRate $run.LearningRate `
        -BiasLearningRate $run.BiasLearningRate `
        -AverageSweeps $run.AverageSweeps
}
