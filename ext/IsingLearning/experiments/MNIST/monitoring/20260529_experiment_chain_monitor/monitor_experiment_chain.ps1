param(
    [int]$PollSeconds = 60
)

$ErrorActionPreference = "Continue"

$Repo = Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..\..\..")
$LogPath = Join-Path $PSScriptRoot "monitor.log"
$AlertPath = Join-Path $PSScriptRoot "alerts.log"

$Experiments = @(
    @{
        Name = "r8_relax_beta"
        Root = Join-Path $Repo "ext\IsingLearning\experiments\MNIST\single-hidden-local-28x28-to-11x11-readout\experiments\current\20260529_mnist_single_hidden_local_r8_relax_beta_grid_e30_32w"
        ExpectedRows = 12
    },
    @{
        Name = "r1_to_r9_relax"
        Root = Join-Path $Repo "ext\IsingLearning\experiments\MNIST\single-hidden-local-28x28-to-11x11-readout\experiments\current\20260529_mnist_single_hidden_local_r1_to_r9_relax_grid_e30_32w"
        ExpectedRows = 18
    },
    @{
        Name = "r8_high_sweeps"
        Root = Join-Path $Repo "ext\IsingLearning\experiments\MNIST\single-hidden-local-28x28-to-11x11-readout\experiments\current\20260529_mnist_single_hidden_local_r8_high_sweeps_e30_32w"
        ExpectedRows = 5
    },
    @{
        Name = "baseline_revisit"
        Root = Join-Path $Repo "ext\IsingLearning\experiments\MNIST\784-120-40-baseline\experiments\current\20260529_mnist_784_120_40_baseline_revisit_gt90_e40_32w"
        ExpectedRows = 3
    }
)

function Write-MonitorLine {
    param([string]$Message)
    "[$(Get-Date -Format s)] $Message" | Out-File -FilePath $LogPath -Append
}

function Write-AlertLine {
    param([string]$Message)
    "[$(Get-Date -Format s)] $Message" | Out-File -FilePath $AlertPath -Append
    Write-MonitorLine "ALERT $Message"
}

function Get-DataRows {
    param([string]$CsvPath)
    if (!(Test-Path $CsvPath)) {
        return 0
    }
    $lineCount = (Get-Content -LiteralPath $CsvPath | Measure-Object -Line).Lines
    return [Math]::Max(0, $lineCount - 1)
}

function Get-LastCsvLine {
    param([string]$CsvPath)
    if (!(Test-Path $CsvPath)) {
        return "none"
    }
    return (Get-Content -LiteralPath $CsvPath -Tail 1)
}

function Check-Experiment {
    param($Experiment)

    $root = $Experiment.Root
    $name = $Experiment.Name
    $summary = Join-Path $root "grid_summary.csv"
    $stdout = Join-Path $root "logs\launcher_stdout.log"
    $stderr = Join-Path $root "logs\launcher_stderr.log"
    $queue = Join-Path $root "logs\queue.log"
    $rows = Get-DataRows $summary
    $last = Get-LastCsvLine $summary

    Write-MonitorLine "$name rows=$rows/$($Experiment.ExpectedRows) last_summary=$last"

    if (Test-Path $queue) {
        $queueTail = Get-Content -LiteralPath $queue -Tail 1
        Write-MonitorLine "$name queue=$queueTail"
    }

    if (Test-Path $stderr) {
        $stderrItem = Get-Item -LiteralPath $stderr
        if ($stderrItem.Length -gt 0) {
            $tail = (Get-Content -LiteralPath $stderr -Tail 5) -join " | "
            Write-AlertLine "$name stderr_nonempty bytes=$($stderrItem.Length) tail=$tail"
        }
    }

    if (Test-Path $stdout) {
        $tail = Get-Content -LiteralPath $stdout -Tail 60
        $errorLines = @($tail | Where-Object { $_ -match "ERROR:|LoadError|UndefVarError|Stacktrace|failed|exception" })
        if ($errorLines.Count -gt 0) {
            Write-AlertLine "$name stdout_error_tail=$(($errorLines -join ' | '))"
        }
    }

    if (Test-Path $summary) {
        $errorRows = @(Import-Csv -LiteralPath $summary | Where-Object { $_.status -eq "error" })
        if ($errorRows.Count -gt 0) {
            Write-AlertLine "$name summary_error_rows=$($errorRows.Count)"
        }
    }

    return ($rows -ge $Experiment.ExpectedRows)
}

Write-MonitorLine "monitor started poll_seconds=$PollSeconds repo=$Repo"

while ($true) {
    try {
        $julia = @(Get-Process julia -ErrorAction SilentlyContinue | Select-Object Id, CPU, StartTime)
        $juliaSummary = if ($julia.Count -eq 0) { "none" } else { ($julia | ForEach-Object { "$($_.Id):cpu=$([Math]::Round($_.CPU, 1))" }) -join "," }
        Write-MonitorLine "julia_processes=$juliaSummary"

        $done = 0
        foreach ($experiment in $Experiments) {
            if (Check-Experiment $experiment) {
                $done += 1
            }
        }

        if ($done -eq $Experiments.Count) {
            Write-MonitorLine "all monitored experiments reached expected row counts; monitor exiting"
            break
        }
    } catch {
        Write-AlertLine "monitor_exception=$($_.Exception.Message)"
    }

    Start-Sleep -Seconds $PollSeconds
}
