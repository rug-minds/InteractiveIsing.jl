$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..\..")
Set-Location $RepoRoot

$Checkpoint = "ext/IsingLearning/experiments/MNIST/baseline_new_nudge/experiments/current/symmetric_nudge_target_weight_grid_v4_20260609/b0p020_lr0p00025_T0p00005_wd0p001_step0p5_pos9p0_neg0p50_priortrue_adaptivefalse/best_checkpoint.bin"
$GridRoot = "ext/IsingLearning/experiments/MNIST/baseline_new_nudge/experiments/current/symmetric_nudge_resume_from_v4_best_20260609"
New-Item -ItemType Directory -Force $GridRoot | Out-Null

$Summary = Join-Path $GridRoot "grid_summary.csv"
"resume_checkpoint,beta,lr,temp,weight_decay,stepsize,positive_target_weight,negative_target_weight,reset_opt_state,status,best_epoch,best_test_accuracy,best_test_loss,final_epoch,final_test_accuracy,final_test_loss,outdir" |
    Set-Content $Summary

$Runs = @(
    @{ name = "resume_lr0p00015_reset"; beta = "0.020"; lr = "0.00015"; temp = "0.00005"; wd = "0.001"; stepsize = "0.5"; posw = "9.0"; negw = "0.50"; reset = "true" },
    @{ name = "resume_lr0p00010_reset"; beta = "0.020"; lr = "0.00010"; temp = "0.00005"; wd = "0.001"; stepsize = "0.5"; posw = "9.0"; negw = "0.50"; reset = "true" },
    @{ name = "resume_lr0p00025_keepadam"; beta = "0.020"; lr = "0.00025"; temp = "0.00005"; wd = "0.001"; stepsize = "0.5"; posw = "9.0"; negw = "0.50"; reset = "false" }
)

$SupervisorLog = Join-Path $GridRoot "grid_supervisor.log"
$Julia = (Get-Command julia).Source
$Script = "ext/IsingLearning/experiments/MNIST/baseline_new_nudge/mnist_784_120_40_softplus_margin_symmetric_nudge_adam.jl"

foreach ($Run in $Runs) {
    $Beta = $Run.beta
    $Lr = $Run.lr
    $Temp = $Run.temp
    $WeightDecay = $Run.wd
    $Stepsize = $Run.stepsize
    $PositiveTargetWeight = $Run.posw
    $NegativeTargetWeight = $Run.negw
    $ResetOptState = $Run.reset

    $RunName = $Run.name
    $OutDir = Join-Path $GridRoot $RunName
    New-Item -ItemType Directory -Force $OutDir | Out-Null

    $env:ISING_MNIST_IF_RESUME_FROM = $Checkpoint
    $env:ISING_MNIST_IF_RESUME_EPOCH = "-1"
    $env:ISING_MNIST_IF_RESET_OPT_STATE_ON_RESUME = $ResetOptState
    $env:ISING_MNIST_IF_EPOCHS = "12"
    $env:ISING_MNIST_IF_TRAIN_PER_CLASS = "1280"
    $env:ISING_MNIST_IF_TEST_PER_CLASS = "100"
    $env:ISING_MNIST_IF_TRAIN_EVAL_PER_CLASS = "100"
    $env:ISING_MNIST_IF_BATCHSIZE = "128"
    $env:ISING_MNIST_IF_WORKERS = "32"
    $env:ISING_MNIST_IF_PROGRESS_EVERY_BATCHES = "25"
    $env:ISING_MNIST_IF_EVAL_EVERY = "1"
    $env:ISING_MNIST_IF_EARLY_STOP_DECLINE_EPOCHS = "2"
    $env:ISING_MNIST_IF_EARLY_STOP_DECLINE_BATCHES = "0"
    $env:ISING_MNIST_IF_EARLY_STOP_MIN_BATCHES = "0"
    $env:ISING_MNIST_IF_SKIP_WORKER_CLOSE = "true"
    $env:ISING_MNIST_IF_SWEEPS = "500"
    $env:ISING_MNIST_IF_BETA = $Beta
    $env:ISING_MNIST_IF_LR = $Lr
    $env:ISING_MNIST_IF_TEMP = $Temp
    $env:ISING_MNIST_IF_NUDGE_TEMP_PEAK = $Temp
    $env:ISING_MNIST_IF_STEPSIZE = $Stepsize
    $env:ISING_MNIST_IF_WEIGHT_DECAY = $WeightDecay
    $env:ISING_MNIST_IF_POSITIVE_TARGET_WEIGHT = $PositiveTargetWeight
    $env:ISING_MNIST_IF_NEGATIVE_TARGET_WEIGHT = $NegativeTargetWeight
    $env:ISING_MNIST_IF_PROJECT_OUTPUT_BIAS_PRIOR = "true"
    $env:ISING_MNIST_IF_ADAPTIVE_WEIGHT_DECAY = "false"
    $env:ISING_MNIST_IF_OUTDIR = $OutDir

    $RunLog = Join-Path $OutDir "run.log"
    $StdoutLog = Join-Path $OutDir "stdout.log"
    $StderrLog = Join-Path $OutDir "stderr.log"
    "START name=$RunName checkpoint=$Checkpoint beta=$Beta lr=$Lr temp=$Temp wd=$WeightDecay reset_opt_state=$ResetOptState outdir=$OutDir $(Get-Date -Format s)" |
        Tee-Object -FilePath $SupervisorLog -Append

    try {
        $Args = @("-t", "32", "--project=ext/IsingLearning", $Script)
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
                "$Checkpoint,$Beta,$Lr,$Temp,$WeightDecay,$Stepsize,$PositiveTargetWeight,$NegativeTargetWeight,$ResetOptState,done,$($Best.epoch),$($Best.test_accuracy),$($Best.test_loss),$($Final.epoch),$($Final.test_accuracy),$($Final.test_loss),$OutDir" |
                    Add-Content $Summary
            } else {
                "$Checkpoint,$Beta,$Lr,$Temp,$WeightDecay,$Stepsize,$PositiveTargetWeight,$NegativeTargetWeight,$ResetOptState,done_no_eval,,,,,,$OutDir" |
                    Add-Content $Summary
            }
        } else {
            "$Checkpoint,$Beta,$Lr,$Temp,$WeightDecay,$Stepsize,$PositiveTargetWeight,$NegativeTargetWeight,$ResetOptState,done_no_csv,,,,,,$OutDir" |
                Add-Content $Summary
        }
    } catch {
        $Message = $_.Exception.Message.Replace(",", ";")
        "$Checkpoint,$Beta,$Lr,$Temp,$WeightDecay,$Stepsize,$PositiveTargetWeight,$NegativeTargetWeight,$ResetOptState,failed:$Message,,,,,,$OutDir" |
            Add-Content $Summary
        "FAILED name=$RunName $Message" |
            Tee-Object -FilePath $SupervisorLog -Append
    }

    "END name=$RunName beta=$Beta lr=$Lr temp=$Temp wd=$WeightDecay reset_opt_state=$ResetOptState $(Get-Date -Format s)" |
        Tee-Object -FilePath $SupervisorLog -Append
}
