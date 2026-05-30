param(
    [int]$WaitPid = 0
)

$ErrorActionPreference = "Stop"

$Repo = "C:\Users\fenje\dev\InteractiveIsing.jl"
$Driver = Join-Path $Repo "ext\IsingLearning\experiments\MNIST\single-hidden-local-28x28-to-11x11-readout\mnist_local_manager_grid.jl"
$Root = Join-Path $Repo "ext\IsingLearning\experiments\MNIST\single-hidden-local-28x28-to-11x11-readout\experiments\current\20260530_mnist_single_hidden_local_r1_to_r10_low_lr_no_output_bias_sweep_grid_e60_32w"
$LaunchLog = Join-Path $Root "launch\launcher_progress.log"

Set-Location $Repo
New-Item -ItemType Directory -Force -Path (Split-Path $LaunchLog) | Out-Null

if ($WaitPid -gt 0) {
    "[$(Get-Date -Format s)] waiting for launcher pid=$WaitPid" | Tee-Object -FilePath $LaunchLog -Append
    Wait-Process -Id $WaitPid -ErrorAction SilentlyContinue
}

$env:ISING_MNIST_PM_PROGRESS = "true"
$env:ISING_MNIST_PM_PROGRESS_BAR = "false"
$env:ISING_MNIST_PM_WORKERS = "32"
$env:ISING_MNIST_PM_BATCHSIZE = "32"
$env:ISING_MNIST_PM_EPOCHS = "60"
$env:ISING_MNIST_PM_TRAIN_PER_CLASS = "100"
$env:ISING_MNIST_PM_TEST_PER_CLASS = "20"
$env:ISING_MNIST_PM_TRAIN_OUTPUT_BIAS = "false"
$env:ISING_MNIST_PM_TARGET_ON = "1.0"
$env:ISING_MNIST_PM_TARGET_OFF = "-1.0"
$env:ISING_MNIST_PM_BETA = "5.0"
$env:ISING_MNIST_PM_LR_W0 = "0.0004"
$env:ISING_MNIST_PM_LR_W12 = "0.0004"
$env:ISING_MNIST_PM_LR_W2O = "0.0004"
$env:ISING_MNIST_PM_LR_B = "0.00004"
$env:ISING_MNIST_PM_PROGRESS_EVERY = "10"

$Sweeps = @(25, 50, 100)
$Radii = 1..10

"[$(Get-Date -Format s)] launcher started root=$Root" | Tee-Object -FilePath $LaunchLog -Append
foreach ($sweep in $Sweeps) {
    $sweepRoot = Join-Path $Root "s$sweep"
    New-Item -ItemType Directory -Force -Path $sweepRoot | Out-Null

    foreach ($radius in $Radii) {
        $name = "r${radius}_s${sweep}_e60_low_lr_nooutbias"
        $runOut = Join-Path $sweepRoot "r${radius}_e60_low_lr_nooutbias"
        New-Item -ItemType Directory -Force -Path $runOut | Out-Null

        $env:ISING_MNIST_PM_NAME = $name
        $env:ISING_MNIST_PM_RADII = "$radius"
        $env:ISING_MNIST_PM_RADIUS = "$radius"
        $env:ISING_MNIST_PM_FREE_SWEEPS = "$sweep"
        $env:ISING_MNIST_PM_NUDGE_SWEEPS = "$sweep"
        $env:ISING_MNIST_PM_OUTDIR = $runOut

        "[$(Get-Date -Format s)] run started name=$name radius=$radius sweeps=$sweep outdir=$runOut" | Tee-Object -FilePath $LaunchLog -Append
        $stdoutPath = Join-Path $runOut "stdout.log"
        $stderrPath = Join-Path $runOut "stderr.log"
        $proc = Start-Process -FilePath "julia.exe" `
            -ArgumentList @("--project=ext\IsingLearning", "-t", "32", $Driver) `
            -NoNewWindow `
            -Wait `
            -PassThru `
            -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath
        $exitCode = $proc.ExitCode
        "[$(Get-Date -Format s)] run finished name=$name exit=$exitCode" | Tee-Object -FilePath $LaunchLog -Append
        if ($exitCode -ne 0) {
            throw "run failed: $name exit=$exitCode"
        }
    }
}
"[$(Get-Date -Format s)] launcher finished" | Tee-Object -FilePath $LaunchLog -Append
