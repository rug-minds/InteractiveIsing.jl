param(
    [int]$WaitPid = 0
)

$ErrorActionPreference = "Stop"

$Repo = "C:\Users\fenje\dev\InteractiveIsing.jl"
$Driver = Join-Path $Repo "ext\IsingLearning\experiments\MNIST\single-hidden-local-28x28-to-11x11-readout\mnist_local_manager_grid.jl"
$Root = Join-Path $Repo "ext\IsingLearning\experiments\MNIST\single-hidden-local-28x28-to-11x11-readout\experiments\current\20260530_mnist_single_hidden_local_r4_rescue_lr_beta_sweep_grid_e80_32w"
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
$env:ISING_MNIST_PM_EPOCHS = "80"
$env:ISING_MNIST_PM_TRAIN_PER_CLASS = "100"
$env:ISING_MNIST_PM_TEST_PER_CLASS = "20"
$env:ISING_MNIST_PM_RADII = "4"
$env:ISING_MNIST_PM_RADIUS = "4"
$env:ISING_MNIST_PM_TRAIN_OUTPUT_BIAS = "false"
$env:ISING_MNIST_PM_TARGET_ON = "1.0"
$env:ISING_MNIST_PM_TARGET_OFF = "-1.0"
$env:ISING_MNIST_PM_PROGRESS_EVERY = "10"

$LearningRates = @(
    @{ W = "0.0002"; B = "0.00002"; Label = "0002" },
    @{ W = "0.0004"; B = "0.00004"; Label = "0004" },
    @{ W = "0.0008"; B = "0.00008"; Label = "0008" }
)
$Sweeps = @(50, 100, 200)
$Betas = @(
    @{ Value = "1.0"; Label = "1p0" },
    @{ Value = "2.5"; Label = "2p5" },
    @{ Value = "5.0"; Label = "5p0" }
)

"[$(Get-Date -Format s)] launcher started root=$Root" | Tee-Object -FilePath $LaunchLog -Append
foreach ($lr in $LearningRates) {
    $env:ISING_MNIST_PM_LR_W0 = $lr.W
    $env:ISING_MNIST_PM_LR_W12 = $lr.W
    $env:ISING_MNIST_PM_LR_W2O = $lr.W
    $env:ISING_MNIST_PM_LR_B = $lr.B
    $lrRoot = Join-Path $Root "lr$($lr.Label)"
    New-Item -ItemType Directory -Force -Path $lrRoot | Out-Null

    foreach ($sweep in $Sweeps) {
        $sweepRoot = Join-Path $lrRoot "s$sweep"
        New-Item -ItemType Directory -Force -Path $sweepRoot | Out-Null

        foreach ($beta in $Betas) {
            $name = "r4_lr$($lr.Label)_s${sweep}_beta$($beta.Label)_e80_nooutbias"
            $runOut = Join-Path (Join-Path $sweepRoot "beta$($beta.Label)") "r4_e80"
            New-Item -ItemType Directory -Force -Path $runOut | Out-Null

            $env:ISING_MNIST_PM_NAME = $name
            $env:ISING_MNIST_PM_FREE_SWEEPS = "$sweep"
            $env:ISING_MNIST_PM_NUDGE_SWEEPS = "$sweep"
            $env:ISING_MNIST_PM_BETA = $beta.Value
            $env:ISING_MNIST_PM_OUTDIR = $runOut

            "[$(Get-Date -Format s)] run started name=$name lr_w=$($lr.W) lr_b=$($lr.B) sweeps=$sweep beta=$($beta.Value) outdir=$runOut" | Tee-Object -FilePath $LaunchLog -Append
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
}
"[$(Get-Date -Format s)] launcher finished" | Tee-Object -FilePath $LaunchLog -Append
