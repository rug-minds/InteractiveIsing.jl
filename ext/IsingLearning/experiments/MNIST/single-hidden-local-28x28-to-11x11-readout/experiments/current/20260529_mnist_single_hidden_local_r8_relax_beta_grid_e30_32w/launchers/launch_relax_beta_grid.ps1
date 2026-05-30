$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Repo = Resolve-Path (Join-Path $Root "..\..\..\..\..\..\..\..")
$JuliaLauncher = Join-Path $PSScriptRoot "launch_relax_beta_grid.jl"
$Stdout = Join-Path $Root "logs\launcher_stdout.log"
$Stderr = Join-Path $Root "logs\launcher_stderr.log"

New-Item -ItemType Directory -Force -Path (Join-Path $Root "logs") | Out-Null

$env:JULIA_NUM_THREADS = "32"
$env:ISING_MNIST_PM_PROGRESS = "true"
$env:ISING_MNIST_PM_PROGRESS_BAR = "false"
$env:ISING_MNIST_PM_DYNAMICS = "metropolis"
Remove-Item Env:\ISING_MNIST_PM_RESUME_CHECKPOINT -ErrorAction SilentlyContinue

Start-Process `
    -FilePath "julia" `
    -ArgumentList @("-t", "32", "--project=.", $JuliaLauncher) `
    -WorkingDirectory $Repo `
    -RedirectStandardOutput $Stdout `
    -RedirectStandardError $Stderr `
    -WindowStyle Hidden

Write-Host "Started relaxation/beta grid."
Write-Host "Root: $Root"
Write-Host "Stdout: $Stdout"
Write-Host "Stderr: $Stderr"
