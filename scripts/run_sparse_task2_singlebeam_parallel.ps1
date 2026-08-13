$ErrorActionPreference = 'Stop'

$pythonExe = 'D:\Anaconda3\envs\radioflow-win\python.exe'
$repoRoot = 'E:\RadioFlow-worktrees\multiconfig-srm-01x'
$datasetRoot = 'E:\datasets\MultiConfigRadiomap'
$manifestRoot = Join-Path $datasetRoot 'manifests'
$heightStats = Join-Path $manifestRoot 'height_stats_train.json'
$runRoot = 'E:\RadioFlow\results\sparse_task2_singlebeam_feature5_samples819'
$arrays = @('8x8', '16x16', '32x32')
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'

if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw "Python environment is missing: $pythonExe"
}
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null

foreach ($array in $arrays) {
    $manifest = Join-Path $manifestRoot ("manifest_singlebeam_feature5_samples819_{0}.jsonl" -f $array)
    if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) {
        throw "Task 2 manifest is missing: $manifest"
    }
    $runDir = Join-Path $runRoot $array
    $stdout = Join-Path $runRoot ("train_{0}_stdout.log" -f $array)
    $stderr = Join-Path $runRoot ("train_{0}_stderr.log" -f $array)
    $arguments = @(
        '-u',
        (Join-Path $repoRoot 'train_sparse_task2.py'),
        '--dataset-root', $datasetRoot,
        '--manifest-path', $manifest,
        '--height-stats-path', $heightStats,
        '--run-root', $runRoot,
        '--array-size', $array,
        '--device', 'cuda:0',
        '--resume', 'auto'
    )
    $process = Start-Process -FilePath $pythonExe `
        -ArgumentList $arguments `
        -WorkingDirectory $repoRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru
    Write-Output ("started Task 2 single-beam {0}: PID {1}; run_dir={2}" -f $array, $process.Id, $runDir)
}
