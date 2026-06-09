$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..\..\..")
$Project = Join-Path $RepoRoot "ext\IsingLearning"
$Script = Join-Path $ScriptDir "mnist_lcl_6x6_window_adam.jl"
$Root = Join-Path $ScriptDir "experiments\current\20260606_003211_lcl_6x6_tangent_beta_temp_grid"
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

function Set-CommonEnv {
    param(
        [Parameter(Mandatory = $true)][string]$Beta,
        [Parameter(Mandatory = $true)][string]$LearningRate,
        [Parameter(Mandatory = $true)][string]$Temp,
        [Parameter(Mandatory = $true)][string]$Sweeps,
        [Parameter(Mandatory = $true)][string]$Epochs,
        [Parameter(Mandatory = $true)][string]$OutDir
    )

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
    $env:ISING_MNIST_IF_TEMP = $Temp
    $env:ISING_MNIST_IF_SWEEPS = $Sweeps
    $env:ISING_MNIST_LCL_WINDOW = "6"
    $env:ISING_MNIST_LCL_STRIDE = "1"
    $env:ISING_MNIST_LCL_TANGENT_NUDGE = "true"
    $env:ISING_MNIST_IF_OUTDIR = $OutDir
}

function Invoke-LclRun {
    param(
        [Parameter(Mandatory = $true)][string]$Beta,
        [Parameter(Mandatory = $true)][string]$LearningRate,
        [Parameter(Mandatory = $true)][string]$Temp,
        [Parameter(Mandatory = $true)][string]$Sweeps,
        [Parameter(Mandatory = $true)][string]$Epochs,
        [string]$ResumeFrom = ""
    )

    $name = "lcl6x6_tangent_b$(Format-Token $Beta)_lr$(Format-Token $LearningRate)_T$(Format-Token $Temp)_s$(Format-Token $Sweeps)_e$Epochs"
    $outdir = Join-Path $Root $name
    Set-CommonEnv -Beta $Beta -LearningRate $LearningRate -Temp $Temp -Sweeps $Sweeps -Epochs $Epochs -OutDir $outdir

    $manifest = Join-Path $Root "manifest.csv"
    if (-not (Test-Path $manifest)) {
        "timestamp,name,beta,lr,temp,sweeps,epochs,train_per_class,test_per_class,outdir,resume_from" | Out-File -FilePath $manifest -Encoding utf8
    }

    if ($ResumeFrom -ne "") {
        $env:ISING_MNIST_IF_RESUME_FROM = (Resolve-Path $ResumeFrom).Path
        $env:ISING_MNIST_IF_RESUME_EPOCH = "-1"
    } else {
        Clear-ResumeEnv
    }

    "$(Get-Date -Format s),$name,$Beta,$LearningRate,$Temp,$Sweeps,$Epochs,100,40,$outdir,$ResumeFrom" | Out-File -FilePath $manifest -Append -Encoding utf8

    $suffix = if ($ResumeFrom -ne "") { "_resume" } else { "" }
    $stdout = Join-Path $Logs "$($name)$($suffix)_stdout.log"
    $stderr = Join-Path $Logs "$($name)$($suffix)_stderr.log"
    "[$(Get-Date -Format s)] START $name beta=$Beta lr=$LearningRate temp=$Temp sweeps=$Sweeps epochs=$Epochs resume=$ResumeFrom" | Tee-Object -FilePath $stdout -Append
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

$resumeRun = Join-Path $Root "lcl6x6_tangent_b3p0_lr0p0001_T0p001_s25_e120"
$resumeCheckpoint = Join-Path $resumeRun "latest_checkpoint.bin.tmp"

Invoke-LclRun -Beta "3.0" -LearningRate "0.0001" -Temp "0.001" -Sweeps "25" -Epochs "120" -ResumeFrom $resumeCheckpoint
Invoke-LclRun -Beta "1.0" -LearningRate "0.0001" -Temp "0.01" -Sweeps "25" -Epochs "120"
Invoke-LclRun -Beta "3.0" -LearningRate "0.00003" -Temp "0.01" -Sweeps "25" -Epochs "120"
