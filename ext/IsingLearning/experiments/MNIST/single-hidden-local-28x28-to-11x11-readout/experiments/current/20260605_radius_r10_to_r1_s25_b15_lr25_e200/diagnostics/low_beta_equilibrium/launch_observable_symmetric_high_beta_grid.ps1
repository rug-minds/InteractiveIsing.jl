$ErrorActionPreference = "Stop"

$diag = Split-Path -Parent $MyInvocation.MyCommand.Path
$script = Join-Path $diag "low_beta_stateavg_training.jl"
$root = Join-Path $diag "observable_symmetric_high_beta_grid"
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

function Invoke-ObservableHighBetaRun {
    param(
        [Parameter(Mandatory = $true)][string]$Beta
    )

    $name = "r8_obs_sym_highbeta_b$(Format-Token $Beta)_Tc0p003_Tr0p3_lr1e-5_avg10_e80"
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
    $env:ISING_MNIST_PM_COLD_TEMP = "0.003"
    $env:ISING_MNIST_PM_REVERSE_TEMP = "0.3"
    $env:ISING_MNIST_PM_LR_W0 = "0.00001"
    $env:ISING_MNIST_PM_LR_W12 = "0.00001"
    $env:ISING_MNIST_PM_LR_W2O = "0.00001"
    $env:ISING_MNIST_PM_LR_W11 = "0.0"
    $env:ISING_MNIST_PM_LR_W22 = "0.0"
    $env:ISING_MNIST_PM_LR_WOO = "0.0"
    $env:ISING_MNIST_PM_LR_B = "0.000001"
    $env:ISING_MNIST_PM_PROGRESS = "true"
    $env:ISING_MNIST_PM_PROGRESS_BAR = "false"
    $env:ISING_MNIST_PM_PROGRESS_EVERY = "5"
    $env:ISING_MNIST_PM_RESUME_CHECKPOINT = ""
    $env:ISING_MNIST_STATEAVG_ESTIMATOR = "observable_symmetric"
    $env:ISING_MNIST_STATEAVG_BURNIN_SWEEPS = "25"
    $env:ISING_MNIST_STATEAVG_AVERAGE_SWEEPS = "10"
    $env:ISING_MNIST_STATEAVG_SAMPLE_EVERY_SWEEPS = "1"

    $manifest = Join-Path $root "manifest.csv"
    if (-not (Test-Path $manifest)) {
        "timestamp,name,beta,cold_T,reverse_T,lr_w,lr_b,average_sweeps,outdir" | Out-File -FilePath $manifest -Encoding utf8
    }
    "$(Get-Date -Format s),$name,$Beta,0.003,0.3,0.00001,0.000001,10,$outdir" | Out-File -FilePath $manifest -Append -Encoding utf8

    $stdout = Join-Path $logs "$($name)_stdout.log"
    $stderr = Join-Path $logs "$($name)_stderr.log"
    "[$(Get-Date -Format s)] START $name beta=$Beta" | Tee-Object -FilePath $stdout -Append
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    julia -t 16 --project=ext/IsingLearning $script 1>> $stdout 2>> $stderr
    $exit = $LASTEXITCODE
    $ErrorActionPreference = $oldErrorActionPreference
    "[$(Get-Date -Format s)] END $name exit=$exit" | Tee-Object -FilePath $stdout -Append
    if ($exit -ne 0) {
        throw "Observable high-beta run $name failed with exit code $exit"
    }
}

foreach ($beta in @("3.0", "5.0", "10.0", "15.0")) {
    Invoke-ObservableHighBetaRun -Beta $beta
}
