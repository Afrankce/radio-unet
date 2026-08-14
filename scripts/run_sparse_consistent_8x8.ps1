$ErrorActionPreference = "Stop"
$repo = "E:\RadioFlow-worktrees\multiconfig-srm-01x"
$python = "D:\Anaconda3\envs\radioflow-win\python.exe"
$dataset = "E:\datasets\MultiConfigRadiomap"
$manifest = "$dataset\manifests\manifest_singlebeam_feature5_samples819_8x8.jsonl"
$heightStats = "$dataset\manifests\height_stats_train.json"
$runRoot = "E:\RadioFlow\results\sparse_consistent_abcd_v1"
$logRoot = "$runRoot\logs_8x8"

New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
Set-Location -LiteralPath $repo

function Invoke-LoggedPython {
    param(
        [string[]]$Arguments,
        [string]$LogName
    )
    $logPath = Join-Path $logRoot $LogName
    $output = & $python @Arguments 2>&1
    $exitCode = $LASTEXITCODE
    $output | Tee-Object -FilePath $logPath
    if ($exitCode -ne 0) {
        throw "command failed with exit code ${exitCode}: $LogName"
    }
}

$arms = @(
    "environment_only",
    "concat_fullfm",
    "multiscale_fullfm",
    "multiscale_consistent"
)

foreach ($arm in $arms) {
    Invoke-LoggedPython -LogName "train_$arm.log" -Arguments @(
        "train_sparse_consistent.py",
        "--dataset-root", $dataset,
        "--manifest-path", $manifest,
        "--height-stats-path", $heightStats,
        "--run-root", $runRoot,
        "--array-size", "8x8",
        "--arm", $arm,
        "--device", "cuda:0",
        "--resume", "auto"
    )
}

foreach ($arm in $arms) {
    Invoke-LoggedPython -LogName "evaluate_$arm.log" -Arguments @(
        "evaluate_sparse_consistent.py",
        "--dataset-root", $dataset,
        "--manifest-path", $manifest,
        "--height-stats-path", $heightStats,
        "--run-root", $runRoot,
        "--array-size", "8x8",
        "--arm", $arm,
        "--device", "cuda:0",
        "--case-index", "0"
    )
}

Invoke-LoggedPython -LogName "summary_8x8.log" -Arguments @(
    "summarize_sparse_consistent.py",
    "--run-root", $runRoot,
    "--arrays", "8x8"
)
