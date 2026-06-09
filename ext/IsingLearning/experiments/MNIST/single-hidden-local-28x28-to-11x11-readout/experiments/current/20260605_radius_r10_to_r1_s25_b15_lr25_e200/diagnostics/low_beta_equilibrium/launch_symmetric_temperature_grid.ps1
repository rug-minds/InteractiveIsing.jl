$ErrorActionPreference = "Stop"

$diag = Split-Path -Parent $MyInvocation.MyCommand.Path
$script = Join-Path $diag "low_beta_stateavg_training.jl"
$root = Join-Path $diag "symmetric_temperature_grid"
$logs = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $root, $logs | Out-Null

function Format-Token {
    param([Parameter(Mandatory = $true)][string]$Value)
    return ($Value -replace "\.", "p" -replace "-", "m")
}

function Invoke-SymmetricTempRun {
    param(
        [Parameter(Mandatory = $true)][string]$Beta,
        [Parameter(Mandatory = $true)][string]$ColdT,
        [Parameter(Mandatory = $true)][string]$ReverseT
    )

    $name = "r8_sym_b$(Format-Token $Beta)_Tc$(Format-Token $ColdT)_Tr$(Format-Token $ReverseT)_burn25_avg10_e60"
    $env:ISING_MNIST_PM_NAME = $name
    $env:ISING_MNIST_PM_OUTDIR = Join-Path $root $name
    $env:ISING_MNIST_PM_RADII = "8"
    $env:ISING_MNIST_PM_RADIUS = "8"
    $env:ISING_MNIST_PM_WORKERS = "16"
    $env:ISING_MNIST_PM_EPOCHS = "60"
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
    $env:ISING_MNIST_PM_LR_W0 = "0.000025"
    $env:ISING_MNIST_PM_LR_W12 = "0.000025"
    $env:ISING_MNIST_PM_LR_W2O = "0.000025"
    $env:ISING_MNIST_PM_LR_W11 = "0.0"
    $env:ISING_MNIST_PM_LR_W22 = "0.0"
    $env:ISING_MNIST_PM_LR_WOO = "0.0"
    $env:ISING_MNIST_PM_LR_B = "0.0000025"
    $env:ISING_MNIST_PM_PROGRESS = "true"
    $env:ISING_MNIST_PM_PROGRESS_BAR = "false"
    $env:ISING_MNIST_PM_RESUME_CHECKPOINT = ""
    $env:ISING_MNIST_STATEAVG_ESTIMATOR = "symmetric"
    $env:ISING_MNIST_STATEAVG_BURNIN_SWEEPS = "25"
    $env:ISING_MNIST_STATEAVG_AVERAGE_SWEEPS = "10"
    $env:ISING_MNIST_STATEAVG_SAMPLE_EVERY_SWEEPS = "1"

    $stdout = Join-Path $logs "$($name)_stdout.log"
    $stderr = Join-Path $logs "$($name)_stderr.log"
    "[$(Get-Date -Format s)] START $name beta=$Beta cold_T=$ColdT reverse_T=$ReverseT" | Tee-Object -FilePath $stdout -Append
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    julia -t 16 --project=ext/IsingLearning $script 1>> $stdout 2>> $stderr
    $exit = $LASTEXITCODE
    $ErrorActionPreference = $oldErrorActionPreference
    "[$(Get-Date -Format s)] END $name exit=$exit" | Tee-Object -FilePath $stdout -Append
    if ($exit -ne 0) {
        throw "Symmetric temperature run $name failed with exit code $exit"
    }
}

$betas = @("0.1", "0.25")
$coldTemps = @("0.003", "0.01", "0.03")
$reverseTemps = @("0.3", "1.0")

foreach ($beta in $betas) {
    foreach ($coldT in $coldTemps) {
        foreach ($reverseT in $reverseTemps) {
            Invoke-SymmetricTempRun -Beta $beta -ColdT $coldT -ReverseT $reverseT
        }
    }
}
