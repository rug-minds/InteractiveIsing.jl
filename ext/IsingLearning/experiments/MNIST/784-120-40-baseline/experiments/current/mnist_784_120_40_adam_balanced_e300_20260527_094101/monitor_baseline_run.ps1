param(
    [Parameter(Mandatory = $true)]
    [string]$RunDir,
    [Parameter(Mandatory = $true)]
    [int]$ProcessId,
    [int]$PollSeconds = 60
)

$ErrorActionPreference = "Stop"

$monitorLog = Join-Path $RunDir "monitor.log"
$stdoutLog = Join-Path $RunDir "stdout.log"
$stderrLog = Join-Path $RunDir "stderr.log"
$csvPath = Join-Path $RunDir "mnist_784_120_40_adam.csv"

function Write-MonitorLine {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
    Add-Content -LiteralPath $monitorLog -Value "[$timestamp] $Message"
}

function Last-NonEmptyLine {
    param([string]$Path)
    if (!(Test-Path $Path)) {
        return $null
    }
    $lines = Get-Content -LiteralPath $Path -Tail 40
    for ($idx = $lines.Count - 1; $idx -ge 0; $idx--) {
        if (![string]::IsNullOrWhiteSpace($lines[$idx])) {
            return $lines[$idx]
        }
    }
    return $null
}

Write-MonitorLine "monitor started pid=$ProcessId poll_s=$PollSeconds"

$lastReportedCsvEpoch = $null
$lastReportedEvalEpoch = $null
$lastReportedStdout = $null

while ($true) {
    $proc = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($null -eq $proc) {
        break
    }

    if (Test-Path $csvPath) {
        $rows = Import-Csv -LiteralPath $csvPath
        if ($rows.Count -gt 0) {
            $last = $rows[-1]
            $lastStdout = Last-NonEmptyLine -Path $stdoutLog
            if ($last.test_accuracy -and $last.test_accuracy -ne 'missing') {
                if ($lastReportedEvalEpoch -ne $last.epoch) {
                    Write-MonitorLine "new_eval epoch=$($last.epoch) test_accuracy=$($last.test_accuracy) test_loss=$($last.test_loss)"
                    $lastReportedEvalEpoch = $last.epoch
                    $lastReportedCsvEpoch = $last.epoch
                } elseif ($null -ne $lastStdout -and $lastReportedStdout -ne $lastStdout) {
                    Write-MonitorLine "running last_eval_epoch=$($last.epoch) test_accuracy=$($last.test_accuracy) stdout=`"$lastStdout`""
                    $lastReportedStdout = $lastStdout
                }
            } else {
                if ($lastReportedCsvEpoch -ne $last.epoch) {
                    if ($null -ne $lastStdout) {
                        Write-MonitorLine "running epoch=$($last.epoch) stdout=`"$lastStdout`""
                        $lastReportedStdout = $lastStdout
                    } else {
                        Write-MonitorLine "running epoch=$($last.epoch) training-in-progress"
                    }
                    $lastReportedCsvEpoch = $last.epoch
                } elseif ($null -ne $lastStdout -and $lastReportedStdout -ne $lastStdout) {
                    Write-MonitorLine "running epoch=$($last.epoch) stdout=`"$lastStdout`""
                    $lastReportedStdout = $lastStdout
                }
            }
        } else {
            Write-MonitorLine "running csv-created-but-empty"
        }
    } else {
        $lastStdout = Last-NonEmptyLine -Path $stdoutLog
        if ($null -ne $lastStdout) {
            Write-MonitorLine "running stdout=`"$lastStdout`""
        } else {
            Write-MonitorLine "running no-output-yet"
        }
    }

    Start-Sleep -Seconds $PollSeconds
}

Write-MonitorLine "process exited"

if (Test-Path $csvPath) {
    $rows = Import-Csv -LiteralPath $csvPath
    if ($rows.Count -gt 0) {
        $best = $rows | Sort-Object { [double]$_.test_accuracy } -Descending | Select-Object -First 1
        $last = $rows[-1]
        Write-MonitorLine "final best_epoch=$($best.epoch) best_test_accuracy=$($best.test_accuracy) final_epoch=$($last.epoch) final_test_accuracy=$($last.test_accuracy)"
    } else {
        Write-MonitorLine "final csv-empty"
    }
} else {
    Write-MonitorLine "final csv-missing"
}

$lastErr = Last-NonEmptyLine -Path $stderrLog
if ($null -ne $lastErr) {
    Write-MonitorLine "stderr-last=`"$lastErr`""
}
