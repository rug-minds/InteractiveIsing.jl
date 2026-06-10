$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..\..")
Set-Location $RepoRoot

$GridRoot = "ext/IsingLearning/experiments/MNIST/baseline_new_nudge/experiments/current/scout_grid_lowbeta_lowtemp_20260609"
New-Item -ItemType Directory -Force $GridRoot | Out-Null

$Summary = Join-Path $GridRoot "grid_summary.csv"
"beta,temp,status,best_epoch,best_test_accuracy,best_test_loss,final_epoch,final_test_accuracy,final_test_loss,outdir" |
    Set-Content $Summary

$Betas = @("0.05", "0.1", "0.2", "0.5")
$Temps = @("0.0001", "0.0003")
$SupervisorLog = Join-Path $GridRoot "grid_supervisor.log"

foreach ($Temp in $Temps) {
    foreach ($Beta in $Betas) {
        $RunName = "b$($Beta.Replace('.', 'p'))_T$($Temp.Replace('.', 'p'))_lr0p0015_e12_pc1280"
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
        $env:ISING_MNIST_IF_OUTDIR = $OutDir

        $RunLog = Join-Path $OutDir "run.log"
        "START beta=$Beta temp=$Temp outdir=$OutDir $(Get-Date -Format s)" |
            Tee-Object -FilePath $SupervisorLog -Append

        try {
            $PreviousErrorActionPreference = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            try {
                julia -t 32 --project=ext/IsingLearning ext/IsingLearning/experiments/MNIST/baseline_new_nudge/mnist_784_120_40_softplus_margin_nudge_adam.jl 2>&1 |
                    Tee-Object -FilePath $RunLog
                if ($LASTEXITCODE -ne 0) {
                    throw "Julia exited with code $LASTEXITCODE"
                }
            } finally {
                $ErrorActionPreference = $PreviousErrorActionPreference
            }

            $Csv = Join-Path $OutDir "mnist_784_120_40_softplus_margin_nudge_adam.csv"
            if (Test-Path $Csv) {
                $Rows = Import-Csv $Csv
                $EvalRows = @($Rows | Where-Object { $_.test_accuracy -ne "missing" })
                if ($EvalRows.Count -gt 0) {
                    $Best = $EvalRows | Sort-Object { [double]$_.test_accuracy } -Descending | Select-Object -First 1
                    $Final = $EvalRows | Select-Object -Last 1
                    "$Beta,$Temp,done,$($Best.epoch),$($Best.test_accuracy),$($Best.test_loss),$($Final.epoch),$($Final.test_accuracy),$($Final.test_loss),$OutDir" |
                        Add-Content $Summary
                } else {
                    "$Beta,$Temp,done_no_eval,,,,,,$OutDir" | Add-Content $Summary
                }
            } else {
                "$Beta,$Temp,done_no_csv,,,,,,$OutDir" | Add-Content $Summary
            }
        } catch {
            $Message = $_.Exception.Message.Replace(",", ";")
            "$Beta,$Temp,failed:$Message,,,,,,$OutDir" | Add-Content $Summary
            "FAILED beta=$Beta temp=$Temp $Message" | Tee-Object -FilePath $SupervisorLog -Append
        }

        "END beta=$Beta temp=$Temp $(Get-Date -Format s)" |
            Tee-Object -FilePath $SupervisorLog -Append
    }
}
