$ErrorActionPreference = "Stop"

$repo = "C:\Users\fenje\dev\InteractiveIsing.jl"
$julia = "C:\Users\fenje\.julia\juliaup\julia-1.12.6+0.x64.w64.mingw32\bin\julia.exe"
$diag = Join-Path $repo "ext\IsingLearning\experiments\MNIST\single-hidden-local-28x28-to-11x11-readout\experiments\current\20260605_radius_r10_to_r1_s25_b15_lr25_e200\diagnostics\low_beta_equilibrium"
$readout = Join-Path $diag "low_beta_multi_init_timeavg_readout.jl"
$trace = Join-Path $diag "low_beta_output_trace_logger.jl"
$logs = Join-Path $diag "logs"
$status = Join-Path $logs "status.log"

New-Item -ItemType Directory -Force -Path $logs | Out-Null
Set-Location $repo

function Run-JuliaDiagnostic {
    param(
        [string]$Name,
        [string]$Script
    )

    $stdout = Join-Path $logs "$Name`_stdout.log"
    $stderr = Join-Path $logs "$Name`_stderr.log"
    Add-Content -Path $status -Value "[$(Get-Date -Format s)] START $Name"
    $proc = Start-Process -FilePath $julia -ArgumentList @("-t", "16", "--project=ext/IsingLearning", $Script) -WorkingDirectory $repo -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
    Add-Content -Path $status -Value "[$(Get-Date -Format s)] PID $Name $($proc.Id)"
    Wait-Process -Id $proc.Id
    $exitCode = $proc.ExitCode
    Add-Content -Path $status -Value "[$(Get-Date -Format s)] END $Name exit=$exitCode"
    if ($exitCode -ne 0) {
        throw "$Name failed with exit code $exitCode"
    }
}

$env:ISING_MNIST_LOW_BETA_OUTDIR = $diag
$env:ISING_MNIST_LOW_BETA_PER_CLASS = "1"
$env:ISING_MNIST_LOW_BETA_REPEATS = "1,4"
$env:ISING_MNIST_LOW_BETA_BURNIN_SWEEPS = "25"
$env:ISING_MNIST_LOW_BETA_AVERAGE_SWEEPS = "10"
$env:ISING_MNIST_LOW_BETA_BETAS = "0.05,0.1,0.25,0.5"
$env:ISING_MNIST_LOW_BETA_SAMPLE_EVERY_SWEEPS = "1"
$env:ISING_MNIST_LOW_BETA_SEED = "991337"

Run-JuliaDiagnostic -Name "readout_multi_init_timeavg_screen" -Script $readout

$env:ISING_MNIST_LOW_BETA_TRACE_SAMPLES = "1,2,3"
$env:ISING_MNIST_LOW_BETA_TRACE_REPEATS = "2"
$env:ISING_MNIST_LOW_BETA_TRACE_BURNIN_SWEEPS = "80"
$env:ISING_MNIST_LOW_BETA_TRACE_NUDGE_SWEEPS = "80"
$env:ISING_MNIST_LOW_BETA_TRACE_BETA = "0.1"
$env:ISING_MNIST_LOW_BETA_TRACE_EVERY_SWEEPS = "2"
$env:ISING_MNIST_LOW_BETA_TRACE_SEED = "812377"

Run-JuliaDiagnostic -Name "output_trace_logger_beta0p1" -Script $trace

Add-Content -Path $status -Value "[$(Get-Date -Format s)] LOW-BETA DIAGNOSTICS COMPLETE"
