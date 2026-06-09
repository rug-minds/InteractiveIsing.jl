$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..\..\..")
$Project = Join-Path $RepoRoot "ext\IsingLearning"
$Script = Join-Path $ScriptDir "mnist_lcl_6x6_window_adam.jl"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Root = Join-Path $ScriptDir ("experiments\current\" + $Stamp + "_lcl_6x6_clamp_warm_temperature_followup")
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
        [Parameter(Mandatory = $true)][string]$Temp,
        [Parameter(Mandatory = $true)][string]$Sweeps,
        [Parameter(Mandatory = $true)][string]$Epochs
    )

    $name = "lcl6x6_clamp_warm_b$(Format-Token $Beta)_lr$(Format-Token $LearningRate)_T$(Format-Token $Temp)_s$(Format-Token $Sweeps)_train500_e$Epochs"
    $outdir = Join-Path $Root $name

    $env:ISING_MNIST_IF_WORKERS = "32"
    $env:ISING_MNIST_IF_EPOCHS = $Epochs
    $env:ISING_MNIST_IF_BATCHSIZE = "200"
    $env:ISING_MNIST_IF_TRAIN_PER_CLASS = "500"
    $env:ISING_MNIST_IF_TEST_PER_CLASS = "100"
    $env:ISING_MNIST_IF_TRAIN_EVAL_PER_CLASS = "50"
    $env:ISING_MNIST_IF_EVAL_EVERY = "5"
    $env:ISING_MNIST_IF_HIDDEN = "529"
    $env:ISING_MNIST_IF_OUTPUT_REPLICAS = "1"
    $env:ISING_MNIST_IF_LR = $LearningRate
    $env:ISING_MNIST_IF_BETA = $Beta
    $env:ISING_MNIST_IF_TEMP = $Temp
    $env:ISING_MNIST_IF_SWEEPS = $Sweeps
    $env:ISING_MNIST_IF_WEIGHT_SCALE = "0.005"
    $env:ISING_MNIST_LCL_WINDOW = "6"
    $env:ISING_MNIST_LCL_STRIDE = "1"
    $env:ISING_MNIST_LCL_TANGENT_NUDGE = "false"
    $env:ISING_MNIST_IF_OUTDIR = $outdir
    Remove-Item Env:\ISING_MNIST_IF_RESUME_FROM -ErrorAction SilentlyContinue
    Remove-Item Env:\ISING_MNIST_IF_RESUME_EPOCH -ErrorAction SilentlyContinue

    $manifest = Join-Path $Root "manifest.csv"
    if (-not (Test-Path $manifest)) {
        "timestamp,name,nudge,beta,lr,temp,sweeps,epochs,workers,julia_threads,batchsize,train_per_class,test_per_class,outdir" | Out-File -FilePath $manifest -Encoding utf8
    }
    "$(Get-Date -Format s),$name,clamp,$Beta,$LearningRate,$Temp,$Sweeps,$Epochs,32,32,200,500,100,$outdir" | Out-File -FilePath $manifest -Append -Encoding utf8

    $stdout = Join-Path $Logs "$($name)_stdout.log"
    $stderr = Join-Path $Logs "$($name)_stderr.log"
    "[$(Get-Date -Format s)] START $name beta=$Beta lr=$LearningRate temp=$Temp sweeps=$Sweeps epochs=$Epochs" | Tee-Object -FilePath $stdout -Append
    $proc = Start-Process julia -ArgumentList @("-t", "32", "--project=$Project", $Script) -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -Wait -PassThru
    "[$(Get-Date -Format s)] END $name exit=$($proc.ExitCode)" | Tee-Object -FilePath $stdout -Append
    if ($proc.ExitCode -ne 0) {
        throw "LCL clamp warm-temperature run $name failed with exit code $($proc.ExitCode)"
    }
}

Invoke-LclRun -Beta "1.0" -LearningRate "0.0005" -Temp "0.010" -Sweeps "25" -Epochs "120"
Invoke-LclRun -Beta "1.0" -LearningRate "0.0005" -Temp "0.030" -Sweeps "25" -Epochs "120"
Invoke-LclRun -Beta "3.0" -LearningRate "0.0005" -Temp "0.010" -Sweeps "25" -Epochs "120"
Invoke-LclRun -Beta "3.0" -LearningRate "0.0003" -Temp "0.030" -Sweeps "25" -Epochs "120"
