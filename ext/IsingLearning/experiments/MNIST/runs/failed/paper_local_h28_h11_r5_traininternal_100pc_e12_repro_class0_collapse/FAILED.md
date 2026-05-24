# Failed Run: Exact Old H28-H11 R5 100/Class Repro Attempt

Use of this file: explain why this failed run was kept instead of deleted.

This run attempted to reproduce the old `20260522_local_paper_h28_h11_traininternal_100pc` settings in the cleaned `mnist_local_paper_manager_grid.jl` file.

It was stopped after epoch 6 because it collapsed to predicting digit 0 for every test sample from epoch 1 onward.

Reason kept: the archived working run with the same visible hyperparameters reached `77.5%` on a 20/class test slice. The clean file therefore appears behaviorally different from the old working implementation; this folder documents the failure mode and should be used while comparing the sample-gradient logic.
