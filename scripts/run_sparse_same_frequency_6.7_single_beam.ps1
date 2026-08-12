$ErrorActionPreference = "Stop"

$pythonExe = "D:\Anaconda3\envs\radioflow-win\python.exe"
$datasetRoot = "E:\datasets\MultiConfigRadiomap"
$manifestRoot = "E:\datasets\MultiConfigRadiomap\manifests"
$heightStats = Join-Path $manifestRoot "height_stats_train.json"
$repoRoot = "E:\RadioFlow-worktrees\multiconfig-srm-01x"
$runRoot = "E:\RadioFlow\runs\sparse_srm_6.7ghz_5pct_single_beam"
$variant = "beam_masked"
$arrays = @("8x8", "16x16", "32x32")
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

foreach ($arraySize in $arrays) {
    $manifest = Join-Path $manifestRoot ("manifest_samefreq_6.7ghz_{0}_0deg.jsonl" -f $arraySize)
    $variantRunRoot = Join-Path $runRoot $variant
    & $pythonExe (Join-Path $repoRoot "train_sparse_same_frequency.py") `
        --dataset-root $datasetRoot `
        --manifest-path $manifest `
        --height-stats-path $heightStats `
        --run-root $variantRunRoot `
        --array-size $arraySize `
        --variant $variant `
        --device cuda:0 `
        --resume auto
    if ($LASTEXITCODE -ne 0) {
        throw "sparse training failed for $variant / $arraySize with exit code $LASTEXITCODE"
    }
}
