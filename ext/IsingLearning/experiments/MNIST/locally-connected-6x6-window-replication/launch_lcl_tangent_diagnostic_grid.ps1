$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..\..\..")
$Project = Join-Path $RepoRoot "ext\IsingLearning"
$Script = Join-Path $ScriptDir "mnist_lcl_6x6_window_adam.jl"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Root = Join-Path $ScriptDir ("experiments\current\" + $Stamp + "_lcl_6x6_tangent_diagnostic_grid")
$Logs = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $Root, $Logs | Out-Null

function Format-Token {
    param([Parameter(Mandatory = $true)][string]$Value)
    return ($Value -replace "\.", "p" -replace "-", "m")
}

function Invoke-LclRun {
    param(
        [Parameter(Mandatory = $true)][string]$Beta,
        [Parameter(Mandatory = $true)][string]$LearningRate,
        [Parameter(Mandatory = $true)][string]$Sweeps,
        [Parameter(Mandatory = $true)][string]$Epochs
    )

    $name = "lcl6x6_tangent_b$(Format-Token $Beta)_lr$(Format-Token $LearningRate)_s$(Format-Token $Sweeps)_e$Epochs"
    $outdir = Join-Path $Root $name

    $env:ISING_MNIST_IF_WORKERS = "32"
    $env:ISING_MNIST_IF_EPOCHS = $Epochs
    $env:ISING_MNIST_IF_BATCHSIZE = "100"
    $env:ISING_MNIST_IF_TRAIN_PER_CLASS = "100"
    $env:ISING_MNIST_IF_TEST_PER_CLASS = "40"
    $env:ISING_MNIST_IF_TRAIN_EVAL_PER_CLASS = "20"
    $env:ISING_MNIST_IF_EVAL_EVERY = "5"
    $env:ISING_MNIST_IF_HIDDEN = "529"
    $env:ISING_MNIST_IF_OUTPUT_REPLICAS = "1"
    $env:ISING_MNIST_IF_LR = $LearningRate
    $env:ISING_MNIST_IF_BETA = $Beta
    $env:ISING_MNIST_IF_SWEEPS = $Sweeps
    $env:ISING_MNIST_LCL_WINDOW = "6"
    $env:ISING_MNIST_LCL_STRIDE = "1"
    $env:ISING_MNIST_LCL_TANGENT_NUDGE = "true"
    $env:ISING_MNIST_IF_OUTDIR = $outdir

    $manifest = Join-Path $Root "manifest.csv"
    if (-not (Test-Path $manifest)) {
        "timestamp,name,beta,lr,sweeps,epochs,train_per_class,test_per_class,outdir" | Out-File -FilePath $manifest -Encoding utf8
    }
    "$(Get-Date -Format s),$name,$Beta,$LearningRate,$Sweeps,$Epochs,100,40,$outdir" | Out-File -FilePath $manifest -Append -Encoding utf8

    $stdout = Join-Path $Logs "$($name)_stdout.log"
    $stderr = Join-Path $Logs "$($name)_stderr.log"
    "[$(Get-Date -Format s)] START $name beta=$Beta lr=$LearningRate sweeps=$Sweeps epochs=$Epochs" | Tee-Object -FilePath $stdout -Append
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    julia -t 32 "--project=$Project" $Script 1>> $stdout 2>> $stderr
    $exit = $LASTEXITCODE
    $ErrorActionPreference = $oldErrorActionPreference
    "[$(Get-Date -Format s)] END $name exit=$exit" | Tee-Object -FilePath $stdout -Append
    if ($exit -ne 0) {
        throw "LCL run $name failed with exit code $exit"
    }
}

$runs = @(
    @{ Beta = "0.1"; LearningRate = "0.0001";  Sweeps = "25"; Epochs = "80" },
    @{ Beta = "0.1"; LearningRate = "0.0003";  Sweeps = "25"; Epochs = "80" },
    @{ Beta = "0.3"; LearningRate = "0.0001";  Sweeps = "25"; Epochs = "80" },
    @{ Beta = "0.1"; LearningRate = "0.0001";  Sweeps = "75"; Epochs = "80" },
    @{ Beta = "0.3"; LearningRate = "0.00003"; Sweeps = "75"; Epochs = "80" }
)

foreach ($run in $runs) {
    Invoke-LclRun `
        -Beta $run.Beta `
        -LearningRate $run.LearningRate `
        -Sweeps $run.Sweeps `
        -Epochs $run.Epochs
}
