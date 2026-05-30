param(
    [int]$ExpectedRows = 5
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Repo = Resolve-Path (Join-Path $Root "..\..\..\..\..\..\..\..")
$JuliaLauncher = Join-Path $PSScriptRoot "launch_baseline_revisit_grid.jl"
$Stdout = Join-Path $Root "logs\launcher_stdout.log"
$Stderr = Join-Path $Root "logs\launcher_stderr.log"
$QueueLog = Join-Path $Root "logs\queue.log"
$HighSweepPath = Join-Path $Repo "ext\IsingLearning\experiments\MNIST\single-hidden-local-28x28-to-11x11-readout\experiments\current\20260529_mnist_single_hidden_local_r8_high_sweeps_e30_32w\grid_summary.csv"

New-Item -ItemType Directory -Force -Path (Join-Path $Root "logs") | Out-Null

"[$(Get-Date -Format s)] Waiting for high-sweep summary rows: $ExpectedRows at $HighSweepPath" | Out-File -FilePath $QueueLog -Append
while ($true) {
    if (Test-Path $HighSweepPath) {
        $lineCount = (Get-Content -LiteralPath $HighSweepPath | Measure-Object -Line).Lines
        $dataRows = [Math]::Max(0, $lineCount - 1)
        if ($dataRows -ge $ExpectedRows) {
            "[$(Get-Date -Format s)] High-sweep grid has $dataRows rows; starting baseline revisit." | Out-File -FilePath $QueueLog -Append
            break
        }
        "[$(Get-Date -Format s)] High-sweep rows=$dataRows/$ExpectedRows; still waiting." | Out-File -FilePath $QueueLog -Append
    } else {
        "[$(Get-Date -Format s)] High-sweep summary not present yet; still waiting." | Out-File -FilePath $QueueLog -Append
    }
    Start-Sleep -Seconds 60
}

$env:JULIA_NUM_THREADS = "32"
Remove-Item Env:\ISING_MNIST_IF_RESUME_FROM -ErrorAction SilentlyContinue
Remove-Item Env:\ISING_MNIST_IF_RESUME_EPOCH -ErrorAction SilentlyContinue

Start-Process `
    -FilePath "julia" `
    -ArgumentList @("-t", "32", "--project=.", $JuliaLauncher) `
    -WorkingDirectory $Repo `
    -RedirectStandardOutput $Stdout `
    -RedirectStandardError $Stderr `
    -WindowStyle Hidden

"[$(Get-Date -Format s)] Started baseline revisit. stdout=$Stdout stderr=$Stderr" | Out-File -FilePath $QueueLog -Append
Write-Host "Started baseline revisit."
Write-Host "Root: $Root"
Write-Host "Stdout: $Stdout"
Write-Host "Stderr: $Stderr"
