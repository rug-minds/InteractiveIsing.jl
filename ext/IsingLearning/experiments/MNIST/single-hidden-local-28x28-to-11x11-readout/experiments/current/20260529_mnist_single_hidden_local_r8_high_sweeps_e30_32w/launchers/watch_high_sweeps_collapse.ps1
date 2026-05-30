param(
    [int]$PollSeconds = 60,
    [int]$CollapseWindow = 3,
    [double]$ChanceAccuracy = 0.105,
    [double]$SingleClassFraction = 0.95
)

$ErrorActionPreference = "Continue"

$RunRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Repo = Resolve-Path (Join-Path $RunRoot "..\..\..\..\..\..\..\..")
$Launcher = Join-Path $RunRoot "launchers\launch_high_sweeps_grid.jl"
$Summary = Join-Path $RunRoot "grid_summary.csv"
$Stdout = Join-Path $RunRoot "logs\launcher_stdout.log"
$Stderr = Join-Path $RunRoot "logs\launcher_stderr.log"
$WatchLog = Join-Path $RunRoot "logs\collapse_watchdog.log"
$AllSweeps = @(150, 250, 400, 800, 1200)
$InvariantCulture = [System.Globalization.CultureInfo]::InvariantCulture

function Write-WatchLog {
    param([string]$Message)
    "[$(Get-Date -Format s)] $Message" | Out-File -FilePath $WatchLog -Append
}

function Get-SummarySweeps {
    if (!(Test-Path $Summary)) {
        return @()
    }
    return @(Import-Csv -LiteralPath $Summary | ForEach-Object { [int]$_.sweeps })
}

function Get-ActiveMetricsFile {
    $reported = Get-SummarySweeps
    $files = @(Get-ChildItem -LiteralPath $RunRoot -Recurse -Filter metrics.csv |
        Where-Object {
            $_.FullName -match "\\s(\d+)\\r8_beta" -and
            ($reported -notcontains [int]$Matches[1])
        } |
        Sort-Object LastWriteTime -Descending)
    if ($files.Count -eq 0) {
        return $null
    }
    return $files[0]
}

function Test-CollapsedMetrics {
    param([string]$MetricsPath)
    $rows = @(Import-Csv -LiteralPath $MetricsPath)
    if ($rows.Count -lt $CollapseWindow) {
        return $false
    }
    $tail = @($rows | Where-Object { [int]$_.epoch -ge 1 } | Select-Object -Last $CollapseWindow)
    if ($tail.Count -lt $CollapseWindow) {
        return $false
    }
    foreach ($row in $tail) {
        $acc = [double]$row.test_accuracy
        if ($acc -gt $ChanceAccuracy) {
            return $false
        }
        $counts = @($row.pred_counts -split "-" | ForEach-Object { [int]$_ })
        $total = ($counts | Measure-Object -Sum).Sum
        $max = ($counts | Measure-Object -Maximum).Maximum
        if ($total -le 0 -or ($max / $total) -lt $SingleClassFraction) {
            return $false
        }
    }
    return $true
}

function Append-CollapsedSummary {
    param(
        [int]$Sweeps,
        [string]$MetricsPath
    )
    $reported = Get-SummarySweeps
    if ($reported -contains $Sweeps) {
        return
    }
    $rows = @(Import-Csv -LiteralPath $MetricsPath)
    $last = $rows[-1]
    $best = ($rows | ForEach-Object { [double]$_.best_accuracy } | Measure-Object -Maximum).Maximum
    $outdir = Split-Path -Parent $MetricsPath
    $row = [ordered]@{
        timestamp = Get-Date -Format s
        status = "collapsed_stopped"
        sweeps = $Sweeps
        radius = 8
        beta = "5.0"
        elapsed_seconds = ""
        best_accuracy = $best
        final_test_accuracy = $last.test_accuracy
        final_test_loss = $last.test_loss
        final_train_accuracy = $last.train_accuracy
        final_train_loss = $last.train_loss
        skipped = $last.skipped
        latest_checkpoint = $last.latest_path
        final_checkpoint = ""
        outdir = $outdir
        error = "early stop: collapsed to chance accuracy and single-class predictions"
    }
    $values = @($row.Values | ForEach-Object {
        if ($_ -is [double] -or $_ -is [float] -or $_ -is [decimal]) {
            $_.ToString($InvariantCulture)
        } else {
            [string]$_
        }
    })
    if (!(Test-Path $Summary) -or (Get-Item -LiteralPath $Summary).Length -eq 0) {
        ($row.Keys -join ",") | Out-File -LiteralPath $Summary -Encoding utf8
        ($values -join ",") | Out-File -LiteralPath $Summary -Append -Encoding utf8
    } else {
        ($values -join ",") | Out-File -LiteralPath $Summary -Append -Encoding utf8
    }
    Write-WatchLog "marked collapsed sweep=$Sweeps metrics=$MetricsPath"
}

function Stop-HighSweepLauncher {
    $procs = @(Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -match "julia|julialauncher" -and
            $_.CommandLine -match [regex]::Escape($Launcher)
        })
    foreach ($proc in $procs) {
        Write-WatchLog "stopping launcher pid=$($proc.ProcessId)"
        Stop-Process -Id $proc.ProcessId -Force
    }
}

function Start-RemainingLauncher {
    $reported = Get-SummarySweeps
    $remaining = @($AllSweeps | Where-Object { $reported -notcontains $_ })
    if ($remaining.Count -eq 0) {
        Write-WatchLog "no remaining high-sweep runs"
        return
    }
    $sweepList = ($remaining -join ",")
    $cmd = "`$env:ISING_MNIST_HIGH_SWEEPS='$sweepList'; julia -t 32 --project=. '$Launcher' >> '$Stdout' 2>> '$Stderr'"
    Write-WatchLog "starting remaining high-sweep launcher sweeps=$sweepList"
    Start-Process -FilePath "pwsh.exe" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $cmd) -WorkingDirectory $Repo -WindowStyle Hidden
}

Write-WatchLog "collapse watchdog started poll_seconds=$PollSeconds collapse_window=$CollapseWindow chance=$ChanceAccuracy single_class_fraction=$SingleClassFraction"

while ($true) {
    try {
        $reported = Get-SummarySweeps
        if ($reported.Count -ge $AllSweeps.Count) {
            Write-WatchLog "all high-sweep rows reported; watchdog exiting"
            break
        }
        $metrics = Get-ActiveMetricsFile
        if ($null -eq $metrics) {
            Write-WatchLog "no active metrics file; reported=$($reported -join ',')"
        } elseif ($metrics.FullName -match "\\s(\d+)\\r8_beta") {
            $sweeps = [int]$Matches[1]
            $rows = @(Import-Csv -LiteralPath $metrics.FullName)
            $last = if ($rows.Count -gt 0) { $rows[-1] } else { $null }
            if ($null -ne $last) {
                Write-WatchLog "active sweep=$sweeps epoch=$($last.epoch) acc=$($last.test_accuracy) pred=$($last.pred_counts)"
            }
            if (Test-CollapsedMetrics $metrics.FullName) {
                Append-CollapsedSummary -Sweeps $sweeps -MetricsPath $metrics.FullName
                Stop-HighSweepLauncher
                Start-Sleep -Seconds 5
                Start-RemainingLauncher
            }
        }
    } catch {
        Write-WatchLog "watchdog_exception=$($_.Exception.Message)"
    }
    Start-Sleep -Seconds $PollSeconds
}
