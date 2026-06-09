$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..\..\..")
$Project = Join-Path $RepoRoot "ext\IsingLearning"
$Script = Join-Path $ScriptDir "mnist_lcl_6x6_window_adam.jl"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Root = Join-Path $ScriptDir ("experiments\current\" + $Stamp + "_lcl_6x6_tangent_corrected_input_beta_temp_grid")
$Logs = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $Root, $Logs | Out-Null

function Format-Token {
    param([Parameter(Mandatory = $true)][string]$Value)
    return ($Value -replace "\.", "p" -replace "-", "m")
}

function Clear-ResumeEnv {
    Remove-Item Env:\ISING_MNIST_IF_RESUME_FROM -ErrorAction SilentlyContinue
    Remove-Item Env:\ISING_MNIST_IF_RESUME_EPOCH -ErrorAction SilentlyContinue
}

function Invoke-LclRun {
    param(
        [Parameter(Mandatory = $true)][string]$Beta,
        [Parameter(Mandatory = $true)][string]$LearningRate,
        [Parameter(Mandatory = $true)][string]$Temp,
        [Parameter(Mandatory = $true)][string]$Sweeps,
        [Parameter(Mandatory = $true)][string]$Epochs,
        [string]$TrainPerClass = "500",
        [string]$TestPerClass = "100",
        [string]$BatchSize = "200",
        [string]$WeightScale = "0.005"
    )

    $name = "lcl6x6_tangent_corrected_b$(Format-Token $Beta)_lr$(Format-Token $LearningRate)_T$(Format-Token $Temp)_ws$(Format-Token $WeightScale)_s$(Format-Token $Sweeps)_train$TrainPerClass`_e$Epochs"
    $outdir = Join-Path $Root $name

    $env:ISING_MNIST_IF_WORKERS = "32"
    $env:ISING_MNIST_IF_EPOCHS = $Epochs
    $env:ISING_MNIST_IF_BATCHSIZE = $BatchSize
    $env:ISING_MNIST_IF_TRAIN_PER_CLASS = $TrainPerClass
    $env:ISING_MNIST_IF_TEST_PER_CLASS = $TestPerClass
    $env:ISING_MNIST_IF_TRAIN_EVAL_PER_CLASS = "50"
    $env:ISING_MNIST_IF_EVAL_EVERY = "5"
    $env:ISING_MNIST_IF_HIDDEN = "529"
    $env:ISING_MNIST_IF_OUTPUT_REPLICAS = "1"
    $env:ISING_MNIST_IF_LR = $LearningRate
    $env:ISING_MNIST_IF_BETA = $Beta
    $env:ISING_MNIST_IF_TEMP = $Temp
    $env:ISING_MNIST_IF_SWEEPS = $Sweeps
    $env:ISING_MNIST_IF_WEIGHT_SCALE = $WeightScale
    $env:ISING_MNIST_LCL_WINDOW = "6"
    $env:ISING_MNIST_LCL_STRIDE = "1"
    $env:ISING_MNIST_LCL_TANGENT_NUDGE = "true"
    $env:ISING_MNIST_IF_OUTDIR = $outdir
    Clear-ResumeEnv

    $manifest = Join-Path $Root "manifest.csv"
    if (-not (Test-Path $manifest)) {
        "timestamp,name,beta,lr,temp,weight_scale,sweeps,epochs,workers,julia_threads,batchsize,train_per_class,test_per_class,outdir" | Out-File -FilePath $manifest -Encoding utf8
    }
    "$(Get-Date -Format s),$name,$Beta,$LearningRate,$Temp,$WeightScale,$Sweeps,$Epochs,32,32,$BatchSize,$TrainPerClass,$TestPerClass,$outdir" | Out-File -FilePath $manifest -Append -Encoding utf8

    $stdout = Join-Path $Logs "$($name)_stdout.log"
    $stderr = Join-Path $Logs "$($name)_stderr.log"
    "[$(Get-Date -Format s)] START $name beta=$Beta lr=$LearningRate temp=$Temp weight_scale=$WeightScale sweeps=$Sweeps epochs=$Epochs train_per_class=$TrainPerClass" | Tee-Object -FilePath $stdout -Append
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    julia -t 32 "--project=$Project" $Script 1>> $stdout 2>> $stderr
    $exit = $LASTEXITCODE
    $ErrorActionPreference = $oldErrorActionPreference
    "[$(Get-Date -Format s)] END $name exit=$exit" | Tee-Object -FilePath $stdout -Append
    if ($exit -ne 0) {
        throw "LCL corrected beta/temp run $name failed with exit code $exit"
    }
}

# The previous low-beta grid used an inconsistent 1/beta^2 scale for input-projection
# weights. These runs use the corrected single 1/beta normalization at minibatch flush.
# Warm temperatures are included because too-cold dynamics may reduce useful exploration.
Invoke-LclRun -Beta "0.1" -LearningRate "0.0003" -Temp "0.001" -Sweeps "25" -Epochs "120"
Invoke-LclRun -Beta "0.1" -LearningRate "0.0005" -Temp "0.003" -Sweeps "25" -Epochs "120"
Invoke-LclRun -Beta "0.1" -LearningRate "0.0005" -Temp "0.010" -Sweeps "25" -Epochs "120"
Invoke-LclRun -Beta "0.3" -LearningRate "0.0003" -Temp "0.001" -Sweeps "25" -Epochs "120"
Invoke-LclRun -Beta "0.3" -LearningRate "0.0005" -Temp "0.003" -Sweeps "25" -Epochs "120"
Invoke-LclRun -Beta "0.3" -LearningRate "0.0005" -Temp "0.010" -Sweeps "25" -Epochs "120"
Invoke-LclRun -Beta "1.0" -LearningRate "0.0005" -Temp "0.001" -Sweeps "25" -Epochs "120"
Invoke-LclRun -Beta "1.0" -LearningRate "0.0005" -Temp "0.003" -Sweeps "25" -Epochs "120"
Invoke-LclRun -Beta "3.0" -LearningRate "0.0005" -Temp "0.001" -Sweeps "25" -Epochs "120"
