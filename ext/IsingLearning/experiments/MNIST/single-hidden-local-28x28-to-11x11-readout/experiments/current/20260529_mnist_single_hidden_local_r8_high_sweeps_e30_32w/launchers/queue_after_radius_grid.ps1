param(
    [int]$ExpectedRows = 18
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Repo = Resolve-Path (Join-Path $Root "..\..\..\..\..\..\..\..")
$JuliaLauncher = Join-Path $PSScriptRoot "launch_high_sweeps_grid.jl"
$Stdout = Join-Path $Root "logs\launcher_stdout.log"
$Stderr = Join-Path $Root "logs\launcher_stderr.log"
$QueueLog = Join-Path $Root "logs\queue.log"
$RadiusSummary = Join-Path $Root "..\20260529_mnist_single_hidden_local_r1_to_r9_relax_grid_e30_32w\grid_summary.csv"

New-Item -ItemType Directory -Force -Path (Join-Path $Root "logs") | Out-Null

"[$(Get-Date -Format s)] Waiting for radius grid summary rows: $ExpectedRows at $RadiusSummary" | Out-File -FilePath $QueueLog -Append
while ($true) {
    if (Test-Path $RadiusSummary) {
        $lineCount = (Get-Content -LiteralPath $RadiusSummary | Measure-Object -Line).Lines
        $dataRows = [Math]::Max(0, $lineCount - 1)
        if ($dataRows -ge $ExpectedRows) {
            "[$(Get-Date -Format s)] Radius grid has $dataRows rows; starting high-sweep grid." | Out-File -FilePath $QueueLog -Append
            break
        }
        "[$(Get-Date -Format s)] Radius grid rows=$dataRows/$ExpectedRows; still waiting." | Out-File -FilePath $QueueLog -Append
    } else {
        "[$(Get-Date -Format s)] Radius grid summary not present yet; still waiting." | Out-File -FilePath $QueueLog -Append
    }
    Start-Sleep -Seconds 60
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

"[$(Get-Date -Format s)] Started high-sweep grid. stdout=$Stdout stderr=$Stderr" | Out-File -FilePath $QueueLog -Append
Write-Host "Started high-sweep grid."
Write-Host "Root: $Root"
Write-Host "Stdout: $Stdout"
Write-Host "Stderr: $Stderr"
