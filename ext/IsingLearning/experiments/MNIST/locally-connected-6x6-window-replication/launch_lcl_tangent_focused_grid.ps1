$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..\..\..")
$Project = Join-Path $RepoRoot "ext\IsingLearning"
$Script = Join-Path $ScriptDir "mnist_lcl_6x6_window_adam.jl"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Root = Join-Path $ScriptDir ("experiments\current\" + $Stamp + "_lcl_6x6_tangent_focused_grid")
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
        [string]$OutputReplicas = "1"
    )

    Clear-ResumeEnv
    $name = "lcl6x6_tangent_b$(Format-Token $Beta)_lr$(Format-Token $LearningRate)_T$(Format-Token $Temp)_s$(Format-Token $Sweeps)_rep$OutputReplicas`_e$Epochs"
    $outdir = Join-Path $Root $name

    $env:ISING_MNIST_IF_WORKERS = "32"
    $env:ISING_MNIST_IF_EPOCHS = $Epochs
    $env:ISING_MNIST_IF_BATCHSIZE = "100"
    $env:ISING_MNIST_IF_TRAIN_PER_CLASS = "100"
    $env:ISING_MNIST_IF_TEST_PER_CLASS = "40"
    $env:ISING_MNIST_IF_TRAIN_EVAL_PER_CLASS = "20"
    $env:ISING_MNIST_IF_EVAL_EVERY = "5"
    $env:ISING_MNIST_IF_HIDDEN = "529"
    $env:ISING_MNIST_IF_OUTPUT_REPLICAS = $OutputReplicas
    $env:ISING_MNIST_IF_LR = $LearningRate
    $env:ISING_MNIST_IF_BETA = $Beta
    $env:ISING_MNIST_IF_TEMP = $Temp
    $env:ISING_MNIST_IF_SWEEPS = $Sweeps
    $env:ISING_MNIST_LCL_WINDOW = "6"
    $env:ISING_MNIST_LCL_STRIDE = "1"
    $env:ISING_MNIST_LCL_TANGENT_NUDGE = "true"
    $env:ISING_MNIST_IF_OUTDIR = $outdir

    $manifest = Join-Path $Root "manifest.csv"
    if (-not (Test-Path $manifest)) {
        "timestamp,name,beta,lr,temp,sweeps,epochs,output_replicas,workers,julia_threads,train_per_class,test_per_class,outdir" | Out-File -FilePath $manifest -Encoding utf8
    }
    "$(Get-Date -Format s),$name,$Beta,$LearningRate,$Temp,$Sweeps,$Epochs,$OutputReplicas,32,32,100,40,$outdir" | Out-File -FilePath $manifest -Append -Encoding utf8

    $stdout = Join-Path $Logs "$($name)_stdout.log"
    $stderr = Join-Path $Logs "$($name)_stderr.log"
    "[$(Get-Date -Format s)] START $name beta=$Beta lr=$LearningRate temp=$Temp sweeps=$Sweeps epochs=$Epochs output_replicas=$OutputReplicas" | Tee-Object -FilePath $stdout -Append
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    julia -t 32 "--project=$Project" $Script 1>> $stdout 2>> $stderr
    $exit = $LASTEXITCODE
    $ErrorActionPreference = $oldErrorActionPreference
    "[$(Get-Date -Format s)] END $name exit=$exit" | Tee-Object -FilePath $stdout -Append
    if ($exit -ne 0) {
        throw "LCL focused run $name failed with exit code $exit"
    }
}

$runs = @(
    @{ Beta = "3.0"; LearningRate = "0.0001";  Temp = "0.001"; Sweeps = "25"; Epochs = "200"; OutputReplicas = "1" },
    @{ Beta = "3.0"; LearningRate = "0.0001";  Temp = "0.001"; Sweeps = "50"; Epochs = "200"; OutputReplicas = "1" },
    @{ Beta = "3.0"; LearningRate = "0.0003";  Temp = "0.001"; Sweeps = "25"; Epochs = "160"; OutputReplicas = "1" },
    @{ Beta = "5.0"; LearningRate = "0.00003"; Temp = "0.001"; Sweeps = "25"; Epochs = "160"; OutputReplicas = "1" },
    @{ Beta = "5.0"; LearningRate = "0.0001";  Temp = "0.001"; Sweeps = "25"; Epochs = "160"; OutputReplicas = "1" },
    @{ Beta = "3.0"; LearningRate = "0.0001";  Temp = "0.001"; Sweeps = "25"; Epochs = "200"; OutputReplicas = "4" }
)

foreach ($run in $runs) {
    Invoke-LclRun `
        -Beta $run.Beta `
        -LearningRate $run.LearningRate `
        -Temp $run.Temp `
        -Sweeps $run.Sweeps `
        -Epochs $run.Epochs `
        -OutputReplicas $run.OutputReplicas
}
