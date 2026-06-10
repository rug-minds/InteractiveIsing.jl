$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..\..")
Set-Location $RepoRoot

$GridRoot = "ext/IsingLearning/experiments/MNIST/baseline_new_nudge/experiments/current/focus_grid_weight_decay_20260609"
New-Item -ItemType Directory -Force $GridRoot | Out-Null

$Summary = Join-Path $GridRoot "grid_summary.csv"
"beta,temp,weight_decay,adaptive,adaptive_w_norm,adaptive_w_input_norm,adaptive_gain,adaptive_max,status,best_epoch,best_test_accuracy,best_test_loss,final_epoch,final_test_accuracy,final_test_loss,outdir" |
    Set-Content $Summary

$Runs = @(
    @{ beta = "0.1"; temp = "0.00005"; wd = "0.0003"; adaptive = "false"; wnorm = "2.0"; winorm = "5.0"; gain = "0.0005"; max = "0.003" },
    @{ beta = "0.1"; temp = "0.00005"; wd = "0.001";  adaptive = "false"; wnorm = "2.0"; winorm = "5.0"; gain = "0.0005"; max = "0.003" },
    @{ beta = "0.1"; temp = "0.00005"; wd = "0.003";  adaptive = "false"; wnorm = "2.0"; winorm = "5.0"; gain = "0.0005"; max = "0.003" },
    @{ beta = "0.2"; temp = "0.00005"; wd = "0.0003"; adaptive = "false"; wnorm = "2.0"; winorm = "5.0"; gain = "0.0005"; max = "0.003" },
    @{ beta = "0.2"; temp = "0.00005"; wd = "0.001";  adaptive = "false"; wnorm = "2.0"; winorm = "5.0"; gain = "0.0005"; max = "0.003" },
    @{ beta = "0.2"; temp = "0.00005"; wd = "0.003";  adaptive = "false"; wnorm = "2.0"; winorm = "5.0"; gain = "0.0005"; max = "0.003" },
    @{ beta = "0.1"; temp = "0.00005"; wd = "0.0001"; adaptive = "true";  wnorm = "1.8"; winorm = "5.0"; gain = "0.00075"; max = "0.003" },
    @{ beta = "0.2"; temp = "0.00005"; wd = "0.0001"; adaptive = "true";  wnorm = "1.8"; winorm = "5.0"; gain = "0.00075"; max = "0.003" }
)

$SupervisorLog = Join-Path $GridRoot "grid_supervisor.log"
$Julia = (Get-Command julia).Source

foreach ($Run in $Runs) {
    $Beta = $Run.beta
    $Temp = $Run.temp
    $WeightDecay = $Run.wd
    $Adaptive = $Run.adaptive
    $AdaptiveWNorm = $Run.wnorm
    $AdaptiveWInputNorm = $Run.winorm
    $AdaptiveGain = $Run.gain
    $AdaptiveMax = $Run.max

    $RunName = "b$($Beta.Replace('.', 'p'))_T$($Temp.Replace('.', 'p'))_wd$($WeightDecay.Replace('.', 'p'))_adaptive$Adaptive"
    $OutDir = Join-Path $GridRoot $RunName
    New-Item -ItemType Directory -Force $OutDir | Out-Null

    $env:ISING_MNIST_IF_EPOCHS = "12"
    $env:ISING_MNIST_IF_TRAIN_PER_CLASS = "1280"
    $env:ISING_MNIST_IF_TEST_PER_CLASS = "100"
    $env:ISING_MNIST_IF_TRAIN_EVAL_PER_CLASS = "100"
    $env:ISING_MNIST_IF_BATCHSIZE = "128"
    $env:ISING_MNIST_IF_WORKERS = "32"
    $env:ISING_MNIST_IF_PROGRESS_EVERY_BATCHES = "25"
    $env:ISING_MNIST_IF_EVAL_EVERY = "1"
    $env:ISING_MNIST_IF_EARLY_STOP_DECLINE_EPOCHS = "2"
    $env:ISING_MNIST_IF_SWEEPS = "500"
    $env:ISING_MNIST_IF_BETA = $Beta
    $env:ISING_MNIST_IF_TEMP = $Temp
    $env:ISING_MNIST_IF_LR = "0.0015"
    $env:ISING_MNIST_IF_WEIGHT_DECAY = $WeightDecay
    $env:ISING_MNIST_IF_ADAPTIVE_WEIGHT_DECAY = $Adaptive
    $env:ISING_MNIST_IF_ADAPTIVE_W_NORM = $AdaptiveWNorm
    $env:ISING_MNIST_IF_ADAPTIVE_W_INPUT_NORM = $AdaptiveWInputNorm
    $env:ISING_MNIST_IF_ADAPTIVE_DECAY_GAIN = $AdaptiveGain
    $env:ISING_MNIST_IF_ADAPTIVE_MAX_DECAY = $AdaptiveMax
    $env:ISING_MNIST_IF_OUTDIR = $OutDir

    $RunLog = Join-Path $OutDir "run.log"
    $StdoutLog = Join-Path $OutDir "stdout.log"
    $StderrLog = Join-Path $OutDir "stderr.log"
    "START beta=$Beta temp=$Temp wd=$WeightDecay adaptive=$Adaptive outdir=$OutDir $(Get-Date -Format s)" |
        Tee-Object -FilePath $SupervisorLog -Append

    try {
        $Args = @(
            "-t", "32",
            "--project=ext/IsingLearning",
            "ext/IsingLearning/experiments/MNIST/baseline_new_nudge/mnist_784_120_40_softplus_margin_nudge_adam.jl"
        )
        $Proc = Start-Process -FilePath $Julia -ArgumentList $Args -Wait -PassThru -NoNewWindow `
            -RedirectStandardOutput $StdoutLog -RedirectStandardError $StderrLog
        Get-Content $StderrLog, $StdoutLog | Set-Content $RunLog
        if ($Proc.ExitCode -ne 0) {
            throw "Julia exited with code $($Proc.ExitCode)"
        }

        $Csv = Join-Path $OutDir "mnist_784_120_40_softplus_margin_nudge_adam.csv"
        if (Test-Path $Csv) {
            $Rows = Import-Csv $Csv
            $EvalRows = @($Rows | Where-Object { $_.test_accuracy -ne "missing" })
            if ($EvalRows.Count -gt 0) {
                $Best = $EvalRows | Sort-Object { [double]$_.test_accuracy } -Descending | Select-Object -First 1
                $Final = $EvalRows | Select-Object -Last 1
                "$Beta,$Temp,$WeightDecay,$Adaptive,$AdaptiveWNorm,$AdaptiveWInputNorm,$AdaptiveGain,$AdaptiveMax,done,$($Best.epoch),$($Best.test_accuracy),$($Best.test_loss),$($Final.epoch),$($Final.test_accuracy),$($Final.test_loss),$OutDir" |
                    Add-Content $Summary
            } else {
                "$Beta,$Temp,$WeightDecay,$Adaptive,$AdaptiveWNorm,$AdaptiveWInputNorm,$AdaptiveGain,$AdaptiveMax,done_no_eval,,,,,,$OutDir" |
                    Add-Content $Summary
            }
        } else {
            "$Beta,$Temp,$WeightDecay,$Adaptive,$AdaptiveWNorm,$AdaptiveWInputNorm,$AdaptiveGain,$AdaptiveMax,done_no_csv,,,,,,$OutDir" |
                Add-Content $Summary
        }
    } catch {
        $Message = $_.Exception.Message.Replace(",", ";")
        "$Beta,$Temp,$WeightDecay,$Adaptive,$AdaptiveWNorm,$AdaptiveWInputNorm,$AdaptiveGain,$AdaptiveMax,failed:$Message,,,,,,$OutDir" |
            Add-Content $Summary
        "FAILED beta=$Beta temp=$Temp wd=$WeightDecay adaptive=$Adaptive $Message" |
            Tee-Object -FilePath $SupervisorLog -Append
    }

    "END beta=$Beta temp=$Temp wd=$WeightDecay adaptive=$Adaptive $(Get-Date -Format s)" |
        Tee-Object -FilePath $SupervisorLog -Append
}
