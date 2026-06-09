$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$script = Join-Path $root "diagnostics\lcl_stochastic_ep_training.jl"
$series = Join-Path $root "experiments\current\20260608_lcl_6x6_stochastic_ep_large_beta_grid"
$logs = Join-Path $series "logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null

$common = @{
    "JULIA_NUM_THREADS" = "32"
    "ISING_MNIST_IF_WORKERS" = "32"
    "ISING_MNIST_IF_BATCHSIZE" = "200"
    "ISING_MNIST_IF_TRAIN_PER_CLASS" = "500"
    "ISING_MNIST_IF_TEST_PER_CLASS" = "100"
    "ISING_MNIST_IF_TRAIN_EVAL_PER_CLASS" = "100"
    "ISING_MNIST_IF_EVAL_EVERY" = "5"
    "ISING_MNIST_IF_EPOCHS" = "80"
    "ISING_MNIST_IF_SWEEPS" = "25"
    "ISING_MNIST_IF_TEMP" = "0.001"
    "ISING_MNIST_IF_STEPSIZE" = "0.5"
    "ISING_MNIST_IF_WEIGHT_SCALE" = "0.005"
    "ISING_MNIST_LCL_TANGENT_NUDGE" = "true"
    "ISING_MNIST_LCL_STOCHASTIC_SAMPLES" = "8"
}

$runs = @(
    @{ name = "b3p0_lr0p0003_T0p001_s25_k8_e80"; beta = "3.0"; lr = "0.0003" },
    @{ name = "b10p0_lr0p001_T0p001_s25_k8_e80"; beta = "10.0"; lr = "0.001" },
    @{ name = "b30p0_lr0p001_T0p001_s25_k8_e80"; beta = "30.0"; lr = "0.001" },
    @{ name = "b100p0_lr0p003_T0p001_s25_k8_e80"; beta = "100.0"; lr = "0.003" }
)

foreach ($run in $runs) {
    Write-Host "Starting $($run.name)"
    foreach ($key in $common.Keys) {
        [Environment]::SetEnvironmentVariable($key, $common[$key], "Process")
    }
    [Environment]::SetEnvironmentVariable("ISING_MNIST_IF_BETA", $run.beta, "Process")
    [Environment]::SetEnvironmentVariable("ISING_MNIST_IF_LR", $run.lr, "Process")
    [Environment]::SetEnvironmentVariable("ISING_MNIST_IF_OUTDIR", (Join-Path $series $run.name), "Process")

    $stdout = Join-Path $logs "$($run.name)_stdout.log"
    $stderr = Join-Path $logs "$($run.name)_stderr.log"
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    julia -t 32 --project=ext/IsingLearning $script 1> $stdout 2> $stderr
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $oldErrorActionPreference
    if ($exitCode -ne 0) {
        throw "Julia run $($run.name) failed with exit code $exitCode. See $stdout and $stderr."
    }
    Write-Host "Finished $($run.name)"
}
