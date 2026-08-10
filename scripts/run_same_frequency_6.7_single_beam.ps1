$ErrorActionPreference = "Stop"

$pythonExe = "D:\Anaconda3\envs\radioflow-win\python.exe"
$datasetRoot = "E:\datasets\MultiConfigRadiomap"
$manifestRoot = "E:\datasets\MultiConfigRadiomap\manifests"
$heightStats = Join-Path $manifestRoot "height_stats_train.json"
$repoRoot = "E:\RadioFlow-worktrees\multiconfig-srm-01x"
$runRoot = "E:\RadioFlow\runs"

foreach ($arraySize in @("8x8", "16x16", "32x32")) {
    $manifest = Join-Path $manifestRoot ("manifest_samefreq_6.7ghz_{0}_0deg.jsonl" -f $arraySize)
    $runDir = Join-Path $runRoot ("srm_samefreq_6.7_to_6.7_{0}_0deg_lite" -f $arraySize)
    & $pythonExe (Join-Path $repoRoot "train_same_frequency.py") `
        --dataset-root $datasetRoot `
        --manifest-path $manifest `
        --height-stats-path $heightStats `
        --run-root $runDir `
        --array-size $arraySize `
        --model-size lite `
        --device cuda:0 `
        --resume auto
    if ($LASTEXITCODE -ne 0) {
        throw "same-frequency training failed for $arraySize with exit code $LASTEXITCODE"
    }
}
