$workspace = "C:\Users\fenje\dev\InteractiveIsing.jl"
$script = "ext/IsingLearning/experiments/MNIST/two-hidden-local-28x28-to-14x14-readout/mnist_local_paper_manager_grid.jl"
$root = "C:\Users\fenje\dev\InteractiveIsing.jl\ext\IsingLearning\experiments\MNIST\two-hidden-local-28x28-to-14x14-readout\diagnostics\runs\grid_r1_r2_lr_sweeps_20260527_015859"
$configs = @(
    @{ r1 = 8; r2 = 1; lr = "0.012"; lrb = "0.0012"; sweeps = "75"; beta = "5.0" },
    @{ r1 = 8; r2 = 2; lr = "0.012"; lrb = "0.0012"; sweeps = "75"; beta = "5.0" },
    @{ r1 = 8; r2 = 3; lr = "0.012"; lrb = "0.0012"; sweeps = "75"; beta = "5.0" },
    @{ r1 = 8; r2 = 4; lr = "0.016"; lrb = "0.0016"; sweeps = "100"; beta = "5.0" },
    @{ r1 = 8; r2 = 5; lr = "0.016"; lrb = "0.0016"; sweeps = "100"; beta = "5.0" },
    @{ r1 = 10; r2 = 2; lr = "0.012"; lrb = "0.0012"; sweeps = "75"; beta = "5.0" },
    @{ r1 = 10; r2 = 3; lr = "0.012"; lrb = "0.0012"; sweeps = "75"; beta = "5.0" },
    @{ r1 = 10; r2 = 4; lr = "0.016"; lrb = "0.0016"; sweeps = "100"; beta = "5.0" },
    @{ r1 = 10; r2 = 5; lr = "0.016"; lrb = "0.0016"; sweeps = "100"; beta = "5.0" }
)
foreach($cfg in $configs){
    $name = "r1_$($cfg.r1)_r2_$($cfg.r2)_lr$($cfg.lr.Replace('.',''))_s$($cfg.sweeps)"
    $outdir = Join-Path $root $name
    Write-Output "START $name $(Get-Date -Format s)"
    Push-Location $workspace
    try {
        $env:ISING_MNIST_2H_WORKERS = '8'
        $env:ISING_MNIST_2H_EPOCHS = '80'
        $env:ISING_MNIST_2H_BATCHSIZE = '32'
        $env:ISING_MNIST_2H_TRAIN_PER_CLASS = '100'
        $env:ISING_MNIST_2H_TEST_PER_CLASS = '20'
        $env:ISING_MNIST_2H_H1_SIDE = '11'
        $env:ISING_MNIST_2H_H2_SIDE = '5'
        $env:ISING_MNIST_2H_PAIRS = "$($cfg.r1):$($cfg.r2)"
        $env:ISING_MNIST_2H_LR_W0 = $cfg.lr
        $env:ISING_MNIST_2H_LR_W12 = $cfg.lr
        $env:ISING_MNIST_2H_LR_W2O = $cfg.lr
        $env:ISING_MNIST_2H_LR_B = $cfg.lrb
        $env:ISING_MNIST_2H_BETA = $cfg.beta
        $env:ISING_MNIST_2H_FREE_SWEEPS = $cfg.sweeps
        $env:ISING_MNIST_2H_NUDGE_SWEEPS = $cfg.sweeps
        $env:ISING_MNIST_2H_PROGRESS = 'false'
        $env:ISING_MNIST_2H_OUTDIR = $outdir
        julia -t 8 --project=ext/IsingLearning $script
    }
    finally {
        Pop-Location
    }
    Write-Output "END $name $(Get-Date -Format s)"
}

