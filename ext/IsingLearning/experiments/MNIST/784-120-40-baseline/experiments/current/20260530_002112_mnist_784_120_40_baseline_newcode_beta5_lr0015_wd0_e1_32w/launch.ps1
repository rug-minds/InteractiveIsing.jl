$ErrorActionPreference = "Stop"

$Repo = "C:\Users\fenje\dev\InteractiveIsing.jl"
$RunRoot = Join-Path $Repo "ext\IsingLearning\experiments\MNIST\784-120-40-baseline\experiments\current\20260530_002112_mnist_784_120_40_baseline_newcode_beta5_lr0015_wd0_e1_32w"
$Logs = Join-Path $RunRoot "logs"
$Julia = "C:\Users\fenje\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe"

New-Item -ItemType Directory -Force -Path $Logs | Out-Null

$env:ISING_MNIST_IF_WORKERS = "32"
$env:ISING_MNIST_IF_EPOCHS = "1"
$env:ISING_MNIST_IF_BATCHSIZE = "128"
$env:ISING_MNIST_IF_TRAIN_PER_CLASS = "5421"
$env:ISING_MNIST_IF_TEST_PER_CLASS = "892"
$env:ISING_MNIST_IF_TRAIN_EVAL_PER_CLASS = "100"
$env:ISING_MNIST_IF_EVAL_EVERY = "1"
$env:ISING_MNIST_IF_SWEEPS = "500"
$env:ISING_MNIST_IF_BETA = "5.0"
$env:ISING_MNIST_IF_LR = "0.0015"
$env:ISING_MNIST_IF_WEIGHT_DECAY = "0.0"
$env:ISING_MNIST_IF_TEMP = "0.001"
$env:ISING_MNIST_IF_STEPSIZE = "0.5"
$env:ISING_MNIST_IF_SEED = "20260526"
$env:ISING_MNIST_IF_OUTDIR = $RunRoot

$stdoutLog = Join-Path $Logs "stdout.log"
$queueLog = Join-Path $Logs "launcher.log"

"$(Get-Date -Format o) starting new-code beta5 lr0015 wd0 one-epoch baseline run" | Out-File -FilePath $queueLog -Encoding utf8 -Append

Set-Location $Repo
& $Julia -t 32 --project=ext/IsingLearning ext/IsingLearning/experiments/MNIST/784-120-40-baseline/mnist_784_120_40_adam.jl *> $stdoutLog

"$(Get-Date -Format o) exited with code $LASTEXITCODE" | Out-File -FilePath $queueLog -Encoding utf8 -Append
exit $LASTEXITCODE
