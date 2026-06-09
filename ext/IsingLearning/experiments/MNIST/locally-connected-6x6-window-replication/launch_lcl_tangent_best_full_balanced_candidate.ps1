$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..\..\..")
$Project = Join-Path $RepoRoot "ext\IsingLearning"
$Script = Join-Path $ScriptDir "mnist_lcl_6x6_window_adam.jl"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Root = Join-Path $ScriptDir ("experiments\current\" + $Stamp + "_lcl_6x6_tangent_best_full_balanced_candidate")
$Logs = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $Root, $Logs | Out-Null

$name = "lcl6x6_tangent_b3p0_lr0p0005_T0p001_s25_fullbalanced_e200"
$outdir = Join-Path $Root $name

Remove-Item Env:\ISING_MNIST_IF_RESUME_FROM -ErrorAction SilentlyContinue
Remove-Item Env:\ISING_MNIST_IF_RESUME_EPOCH -ErrorAction SilentlyContinue

$env:ISING_MNIST_IF_WORKERS = "32"
$env:ISING_MNIST_IF_EPOCHS = "200"
$env:ISING_MNIST_IF_BATCHSIZE = "200"
$env:ISING_MNIST_IF_TRAIN_PER_CLASS = "5421"
$env:ISING_MNIST_IF_TEST_PER_CLASS = "892"
$env:ISING_MNIST_IF_TRAIN_EVAL_PER_CLASS = "100"
$env:ISING_MNIST_IF_EVAL_EVERY = "5"
$env:ISING_MNIST_IF_HIDDEN = "529"
$env:ISING_MNIST_IF_OUTPUT_REPLICAS = "1"
$env:ISING_MNIST_IF_LR = "0.0005"
$env:ISING_MNIST_IF_BETA = "3.0"
$env:ISING_MNIST_IF_TEMP = "0.001"
$env:ISING_MNIST_IF_SWEEPS = "25"
$env:ISING_MNIST_LCL_WINDOW = "6"
$env:ISING_MNIST_LCL_STRIDE = "1"
$env:ISING_MNIST_LCL_TANGENT_NUDGE = "true"
$env:ISING_MNIST_IF_OUTDIR = $outdir

$manifest = Join-Path $Root "manifest.csv"
"timestamp,name,beta,lr,temp,sweeps,epochs,workers,julia_threads,batchsize,train_per_class,test_per_class,outdir" | Out-File -FilePath $manifest -Encoding utf8
"$(Get-Date -Format s),$name,3.0,0.0005,0.001,25,200,32,32,200,5421,892,$outdir" | Out-File -FilePath $manifest -Append -Encoding utf8

$stdout = Join-Path $Logs "$($name)_stdout.log"
$stderr = Join-Path $Logs "$($name)_stderr.log"
"[$(Get-Date -Format s)] START $name beta=3.0 lr=0.0005 temp=0.001 sweeps=25 epochs=200 train_per_class=5421" | Tee-Object -FilePath $stdout -Append
$oldErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
julia -t 32 "--project=$Project" $Script 1>> $stdout 2>> $stderr
$exit = $LASTEXITCODE
$ErrorActionPreference = $oldErrorActionPreference
"[$(Get-Date -Format s)] END $name exit=$exit" | Tee-Object -FilePath $stdout -Append
if ($exit -ne 0) {
    throw "LCL full balanced candidate failed with exit code $exit"
}
