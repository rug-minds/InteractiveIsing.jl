param(
    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"

$root = git rev-parse --show-toplevel
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Set-Location $root
$stashed = $false

$status = git status --porcelain
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ($status) {
    $timestamp = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    git stash push --include-untracked -m "pre-subtree-pull-processes-$timestamp"
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    $stashed = $true
}

# Pull the requested StatefulAlgorithms.jl branch into the vendored subtree.
git subtree pull --prefix=deps/StatefulAlgorithms https://github.com/f-ij/StatefulAlgorithms.jl.git $Branch --squash -m "Pull deps/StatefulAlgorithms subtree from $Branch"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ($stashed) {
    git stash pop
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
