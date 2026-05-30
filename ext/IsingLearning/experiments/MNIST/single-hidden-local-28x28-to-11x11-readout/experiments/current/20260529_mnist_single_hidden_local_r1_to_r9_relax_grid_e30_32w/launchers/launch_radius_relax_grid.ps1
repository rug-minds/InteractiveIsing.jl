param(
    [int]$WaitForPid = 0
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Repo = Resolve-Path (Join-Path $Root "..\..\..\..\..\..\..\..")
$JuliaLauncher = Join-Path $PSScriptRoot "launch_radius_relax_grid.jl"
$Stdout = Join-Path $Root "logs\launcher_stdout.log"
$Stderr = Join-Path $Root "logs\launcher_stderr.log"
$QueueLog = Join-Path $Root "logs\queue.log"

New-Item -ItemType Directory -Force -Path (Join-Path $Root "logs") | Out-Null

if ($WaitForPid -gt 0) {
    "[$(Get-Date -Format s)] Waiting for PID $WaitForPid before starting radius grid." | Out-File -FilePath $QueueLog -Append
    try {
        Wait-Process -Id $WaitForPid
    } catch {
        "[$(Get-Date -Format s)] Wait target PID $WaitForPid is no longer active." | Out-File -FilePath $QueueLog -Append
    }
}

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

"[$(Get-Date -Format s)] Started radius/relax grid. stdout=$Stdout stderr=$Stderr" | Out-File -FilePath $QueueLog -Append
Write-Host "Started radius/relax grid."
Write-Host "Root: $Root"
Write-Host "Stdout: $Stdout"
Write-Host "Stderr: $Stderr"
