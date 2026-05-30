$workspace = "C:\Users\fenje\dev\InteractiveIsing.jl"
$script = "ext/IsingLearning/experiments/MNIST/single-hidden-local-28x28-to-11x11-readout/mnist_local_manager_grid.jl"
$root = "C:\Users\fenje\dev\InteractiveIsing.jl\ext\IsingLearning\experiments\MNIST\single-hidden-local-28x28-to-11x11-readout\diagnostics\runs\grid_r8_r10_lr_sweeps_20260527_015708"
$configs = @(
    @{ radius = 8; lr = "0.004"; lrb = "0.0004"; sweeps = "75"; beta = "5.0" },
    @{ radius = 8; lr = "0.008"; lrb = "0.0008"; sweeps = "75"; beta = "5.0" },
    @{ radius = 8; lr = "0.012"; lrb = "0.0012"; sweeps = "100"; beta = "5.0" },
    @{ radius = 9; lr = "0.004"; lrb = "0.0004"; sweeps = "75"; beta = "5.0" },
    @{ radius = 9; lr = "0.008"; lrb = "0.0008"; sweeps = "75"; beta = "5.0" },
    @{ radius = 9; lr = "0.012"; lrb = "0.0012"; sweeps = "100"; beta = "5.0" },
    @{ radius = 10; lr = "0.004"; lrb = "0.0004"; sweeps = "75"; beta = "5.0" },
    @{ radius = 10; lr = "0.008"; lrb = "0.0008"; sweeps = "75"; beta = "5.0" },
    @{ radius = 10; lr = "0.012"; lrb = "0.0012"; sweeps = "100"; beta = "5.0" }
)
foreach($cfg in $configs){
    $name = "r$($cfg.radius)_lr$($cfg.lr.Replace('.',''))_b$($cfg.beta.Replace('.',''))_s$($cfg.sweeps)"
    $outdir = Join-Path $root $name
    Write-Output "START $name $(Get-Date -Format s)"
    Push-Location $workspace
    try {
        $env:ISING_MNIST_PM_WORKERS = '8'
        $env:ISING_MNIST_PM_EPOCHS = '80'
        $env:ISING_MNIST_PM_BATCHSIZE = '32'
        $env:ISING_MNIST_PM_TRAIN_PER_CLASS = '100'
        $env:ISING_MNIST_PM_TEST_PER_CLASS = '20'
        $env:ISING_MNIST_PM_RADII = "$($cfg.radius)"
        $env:ISING_MNIST_PM_LR_W0 = $cfg.lr
        $env:ISING_MNIST_PM_LR_W12 = $cfg.lr
        $env:ISING_MNIST_PM_LR_W2O = $cfg.lr
        $env:ISING_MNIST_PM_LR_B = $cfg.lrb
        $env:ISING_MNIST_PM_BETA = $cfg.beta
        $env:ISING_MNIST_PM_FREE_SWEEPS = $cfg.sweeps
        $env:ISING_MNIST_PM_NUDGE_SWEEPS = $cfg.sweeps
        $env:ISING_MNIST_PM_PROGRESS = 'false'
        $env:ISING_MNIST_PM_OUTDIR = $outdir
        julia -t 8 --project=ext/IsingLearning $script
    }
    finally {
        Pop-Location
    }
    Write-Output "END $name $(Get-Date -Format s)"
}

