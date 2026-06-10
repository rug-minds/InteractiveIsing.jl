$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..\..")
Set-Location $RepoRoot

$RunRoot = "ext/IsingLearning/experiments/MNIST/baseline_new_nudge/experiments/current/equilibrium_stepsize_diagnostic_20260609"
New-Item -ItemType Directory -Force $RunRoot | Out-Null

$Summary = Join-Path $RunRoot "diagnostic_summary.csv"
"sweeps,stepsize,beta,temp,weight_decay,status,best_epoch,best_test_accuracy,best_test_loss,final_epoch,final_test_accuracy,final_test_loss,outdir" |
    Set-Content $Summary

$Runs = @(
    @{ sweeps = "1000"; stepsize = "0.75" },
    @{ sweeps = "1000"; stepsize = "1.0" },
    @{ sweeps = "1500"; stepsize = "0.75" }
)

$Julia = (Get-Command julia).Source
$SupervisorLog = Join-Path $RunRoot "diagnostic_supervisor.log"

foreach ($Run in $Runs) {
    $Sweeps = $Run.sweeps
    $Stepsize = $Run.stepsize
    $Beta = "0.1"
    $Temp = "0.00005"
    $WeightDecay = "0.001"

    $RunName = "sweeps$($Sweeps)_step$($Stepsize.Replace('.', 'p'))_b0p1_T0p00005_wd0p001"
    $OutDir = Join-Path $RunRoot $RunName
    New-Item -ItemType Directory -Force $OutDir | Out-Null

    $env:ISING_MNIST_IF_EPOCHS = "10"
    $env:ISING_MNIST_IF_TRAIN_PER_CLASS = "1280"
    $env:ISING_MNIST_IF_TEST_PER_CLASS = "100"
    $env:ISING_MNIST_IF_TRAIN_EVAL_PER_CLASS = "100"
    $env:ISING_MNIST_IF_BATCHSIZE = "128"
    $env:ISING_MNIST_IF_WORKERS = "32"
    $env:ISING_MNIST_IF_PROGRESS_EVERY_BATCHES = "25"
    $env:ISING_MNIST_IF_EVAL_EVERY = "1"
    $env:ISING_MNIST_IF_EARLY_STOP_DECLINE_EPOCHS = "2"
    $env:ISING_MNIST_IF_SWEEPS = $Sweeps
    $env:ISING_MNIST_IF_STEPSIZE = $Stepsize
    $env:ISING_MNIST_IF_BETA = $Beta
    $env:ISING_MNIST_IF_TEMP = $Temp
    $env:ISING_MNIST_IF_LR = "0.0015"
    $env:ISING_MNIST_IF_WEIGHT_DECAY = $WeightDecay
    $env:ISING_MNIST_IF_ADAPTIVE_WEIGHT_DECAY = "false"
    $env:ISING_MNIST_IF_OUTDIR = $OutDir

    $RunLog = Join-Path $OutDir "run.log"
    $StdoutLog = Join-Path $OutDir "stdout.log"
    $StderrLog = Join-Path $OutDir "stderr.log"

    "START sweeps=$Sweeps stepsize=$Stepsize beta=$Beta temp=$Temp wd=$WeightDecay outdir=$OutDir $(Get-Date -Format s)" |
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
                "$Sweeps,$Stepsize,$Beta,$Temp,$WeightDecay,done,$($Best.epoch),$($Best.test_accuracy),$($Best.test_loss),$($Final.epoch),$($Final.test_accuracy),$($Final.test_loss),$OutDir" |
                    Add-Content $Summary
            } else {
                "$Sweeps,$Stepsize,$Beta,$Temp,$WeightDecay,done_no_eval,,,,,,$OutDir" |
                    Add-Content $Summary
            }
        } else {
            "$Sweeps,$Stepsize,$Beta,$Temp,$WeightDecay,done_no_csv,,,,,,$OutDir" |
                Add-Content $Summary
        }
    } catch {
        $Message = $_.Exception.Message.Replace(",", ";")
        "$Sweeps,$Stepsize,$Beta,$Temp,$WeightDecay,failed:$Message,,,,,,$OutDir" |
            Add-Content $Summary
        "FAILED sweeps=$Sweeps stepsize=$Stepsize $Message" |
            Tee-Object -FilePath $SupervisorLog -Append
    }

    "END sweeps=$Sweeps stepsize=$Stepsize $(Get-Date -Format s)" |
        Tee-Object -FilePath $SupervisorLog -Append
}
