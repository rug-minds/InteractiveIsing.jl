$ErrorActionPreference = "Stop"

$diag = Split-Path -Parent $MyInvocation.MyCommand.Path
$script = Join-Path $diag "low_beta_stateavg_training.jl"
$root = Join-Path $diag "stateavg_training_runs"
$logs = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $root, $logs | Out-Null

function Invoke-StateAvgRun {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Beta,
        [Parameter(Mandatory = $true)][string]$LrMain,
        [Parameter(Mandatory = $true)][string]$LrBias
    )

    $env:ISING_MNIST_PM_NAME = $Name
    $env:ISING_MNIST_PM_OUTDIR = Join-Path $root $Name
    $env:ISING_MNIST_PM_RADII = "8"
    $env:ISING_MNIST_PM_RADIUS = "8"
    $env:ISING_MNIST_PM_WORKERS = "16"
    $env:ISING_MNIST_PM_EPOCHS = "15"
    $env:ISING_MNIST_PM_BATCHSIZE = "80"
    $env:ISING_MNIST_PM_JOB_CHUNK_SIZE = "1"
    $env:ISING_MNIST_PM_TRAIN_PER_CLASS = "20"
    $env:ISING_MNIST_PM_TEST_PER_CLASS = "10"
    $env:ISING_MNIST_PM_FREE_READS = "1"
    $env:ISING_MNIST_PM_NUDGE_READS = "1"
    $env:ISING_MNIST_PM_FREE_SWEEPS = "15"
    $env:ISING_MNIST_PM_NUDGE_SWEEPS = "15"
    $env:ISING_MNIST_PM_BETA = $Beta
    $env:ISING_MNIST_PM_LR_W0 = $LrMain
    $env:ISING_MNIST_PM_LR_W12 = $LrMain
    $env:ISING_MNIST_PM_LR_W2O = $LrMain
    $env:ISING_MNIST_PM_LR_W11 = "0.0"
    $env:ISING_MNIST_PM_LR_W22 = "0.0"
    $env:ISING_MNIST_PM_LR_WOO = "0.0"
    $env:ISING_MNIST_PM_LR_B = $LrBias
    $env:ISING_MNIST_PM_PROGRESS = "true"
    $env:ISING_MNIST_PM_PROGRESS_BAR = "false"
    $env:ISING_MNIST_PM_RESUME_CHECKPOINT = ""
    $env:ISING_MNIST_STATEAVG_BURNIN_SWEEPS = "15"
    $env:ISING_MNIST_STATEAVG_AVERAGE_SWEEPS = "5"
    $env:ISING_MNIST_STATEAVG_SAMPLE_EVERY_SWEEPS = "1"

    $stdout = Join-Path $logs "$($Name)_stdout.log"
    $stderr = Join-Path $logs "$($Name)_stderr.log"
    "[$(Get-Date -Format s)] START $Name beta=$Beta lr=$LrMain" | Tee-Object -FilePath $stdout -Append
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    julia -t 16 --project=ext/IsingLearning $script 1>> $stdout 2>> $stderr
    $exit = $LASTEXITCODE
    $ErrorActionPreference = $oldErrorActionPreference
    "[$(Get-Date -Format s)] END $Name exit=$exit" | Tee-Object -FilePath $stdout -Append
    if ($exit -ne 0) {
        throw "State-averaged diagnostic $Name failed with exit code $exit"
    }
}

Invoke-StateAvgRun -Name "r8_stateavg_beta0p10_lr1e-5_burn15_avg5_e15" -Beta "0.1" -LrMain "0.00001" -LrBias "0.000001"
Invoke-StateAvgRun -Name "r8_stateavg_beta0p05_lr5e-6_burn15_avg5_e15" -Beta "0.05" -LrMain "0.000005" -LrBias "0.0000005"
