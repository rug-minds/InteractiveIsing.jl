$ErrorActionPreference = "Stop"

$Repo = "C:\Users\fenje\dev\InteractiveIsing.jl"
$Driver = Join-Path $Repo "ext\IsingLearning\experiments\MNIST\single-hidden-local-28x28-to-11x11-readout\mnist_local_manager_grid.jl"
$Root = Join-Path $Repo "ext\IsingLearning\experiments\MNIST\single-hidden-local-28x28-to-11x11-readout\experiments\current\20260530_mnist_single_hidden_local_r8_no_output_bias_relax_beta_grid_e30_32w"
$LaunchLog = Join-Path $Root "launch\launcher_progress.log"

Set-Location $Repo
New-Item -ItemType Directory -Force -Path (Split-Path $LaunchLog) | Out-Null

$env:ISING_MNIST_PM_PROGRESS = "true"
$env:ISING_MNIST_PM_PROGRESS_BAR = "false"
$env:ISING_MNIST_PM_WORKERS = "32"
$env:ISING_MNIST_PM_BATCHSIZE = "32"
$env:ISING_MNIST_PM_EPOCHS = "30"
$env:ISING_MNIST_PM_TRAIN_PER_CLASS = "100"
$env:ISING_MNIST_PM_TEST_PER_CLASS = "20"
$env:ISING_MNIST_PM_RADII = "8"
$env:ISING_MNIST_PM_RADIUS = "8"
$env:ISING_MNIST_PM_TRAIN_OUTPUT_BIAS = "false"
$env:ISING_MNIST_PM_PROGRESS_EVERY = "10"

$Sweeps = @(25, 50, 100, 150)
$Betas = @(
    @{ Value = "2.5"; Label = "2p5" },
    @{ Value = "5.0"; Label = "5p0" },
    @{ Value = "10.0"; Label = "10p0" }
)

"[$(Get-Date -Format s)] launcher started root=$Root" | Tee-Object -FilePath $LaunchLog -Append
foreach ($sweep in $Sweeps) {
    foreach ($beta in $Betas) {
        $name = "r8_s${sweep}_beta$($beta.Label)_e30_nooutbias"
        $runOut = Join-Path $Root $name
        New-Item -ItemType Directory -Force -Path $runOut | Out-Null

        $env:ISING_MNIST_PM_NAME = $name
        $env:ISING_MNIST_PM_FREE_SWEEPS = "$sweep"
        $env:ISING_MNIST_PM_NUDGE_SWEEPS = "$sweep"
        $env:ISING_MNIST_PM_BETA = $beta.Value
        $env:ISING_MNIST_PM_OUTDIR = $runOut

        "[$(Get-Date -Format s)] run started name=$name sweeps=$sweep beta=$($beta.Value) outdir=$runOut" | Tee-Object -FilePath $LaunchLog -Append
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
