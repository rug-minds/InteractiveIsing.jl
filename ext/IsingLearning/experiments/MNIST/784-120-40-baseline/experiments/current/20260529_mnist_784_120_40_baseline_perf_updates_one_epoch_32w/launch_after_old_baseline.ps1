$ErrorActionPreference = "Stop"

$Repo = "C:\Users\fenje\dev\InteractiveIsing.jl"
$RunRoot = Join-Path $Repo "ext\IsingLearning\experiments\MNIST\784-120-40-baseline\experiments\current\20260529_mnist_784_120_40_baseline_perf_updates_one_epoch_32w"
$Logs = Join-Path $RunRoot "logs"
$OldPid = 1068836

New-Item -ItemType Directory -Force -Path $Logs | Out-Null

$queueLog = Join-Path $Logs "queue.log"
"$(Get-Date -Format o) waiting for old baseline PID $OldPid before updated-code timing run" | Out-File -FilePath $queueLog -Encoding utf8 -Append

$oldProcess = Get-Process -Id $OldPid -ErrorAction SilentlyContinue
if ($null -ne $oldProcess) {
    Wait-Process -Id $OldPid
}

"$(Get-Date -Format o) starting updated-code one-epoch baseline timing" | Out-File -FilePath $queueLog -Encoding utf8 -Append

$env:ISING_MNIST_IF_WORKERS = "32"
$env:ISING_MNIST_IF_EPOCHS = "1"
$env:ISING_MNIST_IF_BATCHSIZE = "128"
$env:ISING_MNIST_IF_TRAIN_PER_CLASS = "5421"
$env:ISING_MNIST_IF_TEST_PER_CLASS = "1"
$env:ISING_MNIST_IF_TRAIN_EVAL_PER_CLASS = "0"
$env:ISING_MNIST_IF_EVAL_EVERY = "1"
$env:ISING_MNIST_IF_SWEEPS = "500"
$env:ISING_MNIST_IF_BETA = "5.0"
$env:ISING_MNIST_IF_LR = "0.0015"
$env:ISING_MNIST_IF_WEIGHT_DECAY = "0.0"
$env:ISING_MNIST_IF_TEMP = "0.001"
$env:ISING_MNIST_IF_STEPSIZE = "0.5"
$env:ISING_MNIST_IF_OUTDIR = $RunRoot

Set-Location $Repo
julia -t 32 --project=ext/IsingLearning ext/IsingLearning/experiments/MNIST/784-120-40-baseline/mnist_784_120_40_adam.jl `
    *> (Join-Path $Logs "updated_timing_stdout.log")

"$(Get-Date -Format o) updated-code timing run exited with code $LASTEXITCODE" | Out-File -FilePath $queueLog -Encoding utf8 -Append
exit $LASTEXITCODE
