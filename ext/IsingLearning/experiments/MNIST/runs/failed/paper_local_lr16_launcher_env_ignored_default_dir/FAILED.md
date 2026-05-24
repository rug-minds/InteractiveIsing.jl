# Failed Run: Launcher Environment Ignored

Use of this file: explains why this generated run was kept under `failed`.

This was meant to test the local paper architecture with 32-sample minibatches and `mean` gradients at roughly 16x the old per-sample learning-rate scale. The PowerShell launcher redirected logs correctly, but the environment variables did not reach Julia, so the script used its default output directory and default run settings.

The partial/default run stayed near chance and was stopped. It is kept only to explain the otherwise confusing default-named folder.
