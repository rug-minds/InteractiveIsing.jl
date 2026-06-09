$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..\..\..")
$Project = Join-Path $RepoRoot "ext\IsingLearning"
$Script = Join-Path $ScriptDir "mnist_lcl_6x6_window_adam.jl"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Root = Join-Path $ScriptDir ("experiments\current\" + $Stamp + "_lcl_6x6_tangent_best_longrun")
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
        [string]$TrainPerClass = "100",
        [string]$TestPerClass = "40",
        [string]$BatchSize = "100",
        [string]$ResumeFrom = ""
    )

    $resumeTag = if ($ResumeFrom -ne "") { "_resume" } else { "" }
    $name = "lcl6x6_tangent_b$(Format-Token $Beta)_lr$(Format-Token $LearningRate)_T$(Format-Token $Temp)_s$(Format-Token $Sweeps)_train$TrainPerClass$resumeTag`_e$Epochs"
    $outdir = Join-Path $Root $name

    $env:ISING_MNIST_IF_WORKERS = "32"
    $env:ISING_MNIST_IF_EPOCHS = $Epochs
    $env:ISING_MNIST_IF_BATCHSIZE = $BatchSize
    $env:ISING_MNIST_IF_TRAIN_PER_CLASS = $TrainPerClass
    $env:ISING_MNIST_IF_TEST_PER_CLASS = $TestPerClass
    $env:ISING_MNIST_IF_TRAIN_EVAL_PER_CLASS = ([Math]::Min([int]$TrainPerClass, 20)).ToString()
    $env:ISING_MNIST_IF_EVAL_EVERY = "5"
    $env:ISING_MNIST_IF_HIDDEN = "529"
    $env:ISING_MNIST_IF_OUTPUT_REPLICAS = "1"
    $env:ISING_MNIST_IF_LR = $LearningRate
    $env:ISING_MNIST_IF_BETA = $Beta
    $env:ISING_MNIST_IF_TEMP = $Temp
    $env:ISING_MNIST_IF_SWEEPS = $Sweeps
    $env:ISING_MNIST_LCL_WINDOW = "6"
    $env:ISING_MNIST_LCL_STRIDE = "1"
    $env:ISING_MNIST_LCL_TANGENT_NUDGE = "true"
    $env:ISING_MNIST_IF_OUTDIR = $outdir

    if ($ResumeFrom -ne "") {
        $env:ISING_MNIST_IF_RESUME_FROM = (Resolve-Path $ResumeFrom).Path
        $env:ISING_MNIST_IF_RESUME_EPOCH = "-1"
    } else {
        Clear-ResumeEnv
    }

    $manifest = Join-Path $Root "manifest.csv"
    if (-not (Test-Path $manifest)) {
        "timestamp,name,beta,lr,temp,sweeps,epochs,workers,julia_threads,batchsize,train_per_class,test_per_class,outdir,resume_from" | Out-File -FilePath $manifest -Encoding utf8
    }
    "$(Get-Date -Format s),$name,$Beta,$LearningRate,$Temp,$Sweeps,$Epochs,32,32,$BatchSize,$TrainPerClass,$TestPerClass,$outdir,$ResumeFrom" | Out-File -FilePath $manifest -Append -Encoding utf8

    $stdout = Join-Path $Logs "$($name)_stdout.log"
    $stderr = Join-Path $Logs "$($name)_stderr.log"
    "[$(Get-Date -Format s)] START $name beta=$Beta lr=$LearningRate temp=$Temp sweeps=$Sweeps epochs=$Epochs train_per_class=$TrainPerClass resume=$ResumeFrom" | Tee-Object -FilePath $stdout -Append
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    julia -t 32 "--project=$Project" $Script 1>> $stdout 2>> $stderr
    $exit = $LASTEXITCODE
    $ErrorActionPreference = $oldErrorActionPreference
    "[$(Get-Date -Format s)] END $name exit=$exit" | Tee-Object -FilePath $stdout -Append
    if ($exit -ne 0) {
        throw "LCL long-run $name failed with exit code $exit"
    }
}

$bestFollowupRun = Join-Path $ScriptDir "experiments\current\20260606_011635_lcl_6x6_tangent_followup_grid\lcl6x6_tangent_b3p0_lr0p0005_T0p001_s25_rep1_e220"
$bestFollowupFinal = Join-Path $bestFollowupRun "final_checkpoint.bin"

Invoke-LclRun -Beta "3.0" -LearningRate "0.0005" -Temp "0.001" -Sweeps "25" -Epochs "500" -ResumeFrom $bestFollowupFinal
Invoke-LclRun -Beta "3.0" -LearningRate "0.0007" -Temp "0.001" -Sweeps "25" -Epochs "320"
Invoke-LclRun -Beta "3.0" -LearningRate "0.0005" -Temp "0.001" -Sweeps "50" -Epochs "320"
Invoke-LclRun -Beta "3.0" -LearningRate "0.0005" -Temp "0.001" -Sweeps "25" -Epochs "120" -TrainPerClass "500" -TestPerClass "100" -BatchSize "200"
